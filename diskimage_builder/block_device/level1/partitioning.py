# Copyright 2016 Andreas Florath (andreas@florath.net)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import logging
import os

from diskimage_builder.block_device.exception import \
    BlockDeviceSetupException
from diskimage_builder.block_device.level1.mbr import MBR
from diskimage_builder.block_device.plugin import NodeBase
from diskimage_builder.block_device.plugin import PluginBase
from diskimage_builder.block_device.utils import exec_sudo
from diskimage_builder.block_device.utils import parse_abs_size_spec
from diskimage_builder.block_device.utils import parse_rel_size_spec


logger = logging.getLogger(__name__)

#
# We create a PartitionTableNode as the root of everything, which has
# a dependency to the device it is labeling.  This node then depends
# on all PartitionNodes it describes; additionally those
# PartitionNodes are ordered amongst themselves too.
#
#                       loop device
#                            |
#                            v
#                   PartitionTableNode
#                            |
#         +------------------+----------------+
#         |                  |                |
#         v                  v                v
#        root -----------> second ---------> third  (ParitionNode)
#         |                  |
#         v                  v
#    mount, mkfs etc...     ...
#
# So note that the PartitonNodes are really just place-holders for
# dependencies; the actual partition table is only created once by
# PartitionTableNode.create() (and similarly cleaned up by umount)
#


class PartitionTableNode(NodeBase):

    def __init__(self, name, config, state, base, label, partitions):
        '''A partition table

        Arguments:
        :param name: A symbolic name for this node
        :param config: "partitioning" configuration entry
        :param state: global state pointer
        :param base: the parent device to make partition table on
        :param label: the type of partition table to make
        :param partitions: List of PartitionNode objects to place in this table
        '''
        super(PartitionTableNode, self).__init__(name, state)
        self.base = base
        self.label = label
        self.partitions = partitions

        # It is VERY important to get the alignment correct. If this
        # is not correct, the disk performance might be very poor.
        # Example: In some tests a 'off by one' leads to a write
        # performance of 30% compared to a correctly aligned
        # partition.
        # The problem for DIB is, that it cannot assume that the host
        # system uses the same IO sizes as the target system,
        # therefore here a fixed approach (as used in all modern
        # systems with large disks) is used.  The partitions are
        # aligned to 1MiB (which are about 2048 times 512 bytes
        # blocks)
        self.align = 1024 * 1024  # 1MiB as default
        if 'align' in config:
            self.align = parse_abs_size_spec(config['align'])

    def get_edges(self):
        # we depend on the underlying device
        edge_from = [self.base]
        # all partitions for this table should depend on us
        edge_to = [p.name for p in self.partitions]
        return (edge_from, edge_to)

    def _create_mbr(self):
        """Create partitions with MBR"""
        with MBR(self.image_path, self.disk_size, self.align) as part_impl:
            for part_cfg in self.partitions:
                part_name = part_cfg.get_name()
                part_bootflag = PartitionNode.flag_boot \
                                in part_cfg.get_flags()
                part_primary = PartitionNode.flag_primary \
                               in part_cfg.get_flags()
                part_size = part_cfg.get_size()
                part_free = part_impl.free()
                part_type = part_cfg.get_type()
                logger.debug("Not partitioned space [%d]", part_free)
                part_size = parse_rel_size_spec(part_size,
                                                part_free)[1]
                part_no \
                    = part_impl.add_partition(part_primary, part_bootflag,
                                              part_size, part_type)
                logger.debug("Create partition [%s] [%d]",
                             part_name, part_no)

                # We're going to mount all partitions with kpartx
                # below once we're done.  So the device this partition
                # will be seen at becomes "/dev/mapper/loop0pX"
                assert self.device_path[:5] == "/dev/"
                partition_device_name = "/dev/mapper/%sp%d" % \
                                        (self.device_path[5:], part_no)
                self.state['blockdev'][part_name] \
                    = {'device': partition_device_name}

    def _create_gpt(self):
        """Create partitions with GPT"""

        cmd = ['sgdisk', self.image_path]

        # This padding gives us a little room for rounding so we don't
        # go over the end of the disk
        disk_free = self.disk_size - (2048 * 1024)
        pnum = 1

        for p in self.partitions:
            args = {}
            args['pnum'] = pnum
            args['name'] = '"%s"' % p.get_name()
            args['type'] = '%s' % p.get_type()

            # convert from a relative/string size to bytes
            size = parse_rel_size_spec(p.get_size(), disk_free)[1]

            # We keep track in bytes, but specify things to sgdisk in
            # megabytes so it can align on sensible boundaries. And
            # create partitions right after previous so no need to
            # calculate start/end - just size.
            assert size <= disk_free
            args['size'] = size // (1024 * 1024)

            new_cmd = ("-n {pnum}:0:+{size}M -t {pnum}:{type} "
                       "-c {pnum}:{name}".format(**args))
            cmd.extend(new_cmd.strip().split(' '))

            # Fill the state; we mount all partitions with kpartx
            # below once we're done.  So the device this partition
            # will be seen at becomes "/dev/mapper/loop0pX"
            assert self.device_path[:5] == "/dev/"
            device_name = "/dev/mapper/%sp%d" % (self.device_path[5:], pnum)
            self.state['blockdev'][p.get_name()] \
                = {'device': device_name}

            disk_free = disk_free - size
            pnum = pnum + 1
            logger.debug("Partition %s added, %s remaining in disk",
                         pnum, disk_free)

        logger.debug("cmd: %s", ' '.join(cmd))
        exec_sudo(cmd)

    def _size_of_block_dev(self, dev):
        with open(dev, "r") as fd:
            fd.seek(0, 2)
            return fd.tell()

    def create(self):
        # the raw file on disk
        self.image_path = self.state['blockdev'][self.base]['image']
        # the /dev/loopX device of the parent
        self.device_path = self.state['blockdev'][self.base]['device']
        # underlying size
        self.disk_size = self._size_of_block_dev(self.image_path)

        logger.info("Creating partition on [%s] [%s]",
                    self.base, self.image_path)

        assert self.label in ('mbr', 'gpt')

        if self.label == 'mbr':
            self._create_mbr()
        elif self.label == 'gpt':
            self._create_gpt()

        # "saftey sync" to make sure the partitions are written
        exec_sudo(["sync"])

        # now all the partitions are created, get device-mapper to
        # mount them
        if not os.path.exists("/.dockerenv"):
            exec_sudo(["kpartx", "-avs", self.device_path])
        else:
            # If running inside Docker, make our nodes manually,
            # because udev will not be working. kpartx cannot run in
            # sync mode in docker.
            exec_sudo(["kpartx", "-av", self.device_path])
            exec_sudo(["dmsetup", "--noudevsync", "mknodes"])

        return

    def umount(self):
        exec_sudo(["kpartx", "-d",
                   self.state['blockdev'][self.base]['device']])

    def cleanup(self):
        pass


class PartitionNode(NodeBase):
    flag_boot = 1
    flag_primary = 2

    def __init__(self, config, state, label, prev_partition):
        '''An individual partition

        Argments:
        :param config: individual partition config entry
        :param state: global state reference
        :param label: the partition label type ('mbr' or 'efi')
        :param prev_partition: link to previous PartitionNode for ordering
        '''
        super(PartitionNode, self).__init__(config['name'], state)

        self.base = config['base']
        self.label = label
        self.prev_partition = prev_partition

        # filter out some MBR only options for clarity
        if self.label == 'gpt':
            if 'flags' in config and 'primary' in config['flags']:
                raise BlockDeviceSetupException(
                    "Primary flag not supported for GPT partitions")

        self.flags = set()
        if 'flags' in config:
            for f in config['flags']:
                if f == 'boot':
                    self.flags.add(self.flag_boot)
                elif f == 'primary':
                    self.flags.add(self.flag_primary)
                else:
                    raise BlockDeviceSetupException("Unknown flag: %s" % f)

        if 'size' not in config:
            raise BlockDeviceSetupException("No size in partition" % self.name)
        self.size = config['size']

        if self.label == 'gpt':
            self.ptype = str(config['type']) if 'type' in config else '8300'
        elif self.label == 'mbr':
            self.ptype = int(config['type'], 16) if 'type' in config else 0x83

    def get_flags(self):
        return self.flags

    def get_size(self):
        return self.size

    def get_type(self):
        return self.ptype

    def get_edges(self):
        # The parent PartitionTableNode will depend on all partitions.
        # We just need to keep a dependency chain between partitions
        # so they stay ordered.
        edge_from = []
        edge_to = []
        if self.prev_partition is not None:
            edge_from.append(self.prev_partition.name)
        return (edge_from, edge_to)

    # Note the parent PartitionTableNode has done all the work
    # of setting up the actual partition table.  These nodes
    # are just place-holders for dependency purposes.
    def create(self):
        pass

    def umount(self):
        pass

    def cleanup(self):
        pass


class Partitioning(PluginBase):
    '''The partitioning plugin'''
    def __init__(self, config, default_config, state):
        logger.debug("Creating Partitioning object; config [%s]", config)
        super(Partitioning, self).__init__()

        # Parameter check
        if 'base' not in config:
            raise BlockDeviceSetupException("Partitioning config needs 'base'")
        base = config['base']

        if 'partitions' not in config:
            raise BlockDeviceSetupException(
                "Partitioning config needs 'partitions'")

        if 'label' not in config:
            raise BlockDeviceSetupException(
                "Partitioning config needs 'label'")
        label = config['label']
        if label not in ("mbr", "gpt"):
            raise BlockDeviceSetupException("Label must be 'mbr' or 'gpt'")

        self.partitions = []
        prev_partition = None

        for part_cfg in config['partitions']:
            np = PartitionNode(part_cfg, state, label, prev_partition)
            self.partitions.append(np)
            prev_partition = np

        self.table = PartitionTableNode('%s_%s' % (label, base), config,
                                        state, base, label, self.partitions)

    def get_nodes(self):
        # return the root node and list of partitions
        return [self.table] + self.partitions
