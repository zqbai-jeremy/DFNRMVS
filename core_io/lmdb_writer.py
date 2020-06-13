import os
import lmdb
import shutil
import numpy as np

class LMDBWriter:

    """ Write the dataset to LMDB database
    """

    ''' Variables
    '''
    __key_counts__ = 0

    # LMDB environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Path
    lmdb_path = None

    ''' Functions
    '''
    def __init__(self, lmdb_path, auto_start=True):
        self.lmdb_path = lmdb_path
        self.__del_and_create__(lmdb_path)
        if auto_start is True:
            self.__start_session__()

    def __del__(self):
        self.close_session()

    def __del_and_create__(self, lmdb_path):
        """
        Delete the exist lmdb database and create new lmdb database.
        """
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.mkdir(lmdb_path)

    def __start_session__(self):
        self.__lmdb_env__ = lmdb.Environment(self.lmdb_path, map_size=1099511627776)
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True)

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_txn__.commit()
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None
            self.__lmdb_txn__ = None

    def write_str(self, key, str):
        """
        Write the str data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), str)
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)

    def write_array(self, key, array):
        """
        Write the array data to the LMDB
        :param key: key in string type
        :param array: array data
        """
        # Put to lmdb
        self.__key_counts__ += 1
        self.__lmdb_txn__.put(key.encode(), array.tostring())
        if self.__key_counts__ % 10000 == 0:
            self.__lmdb_txn__.commit()
            self.__lmdb_txn__ = self.__lmdb_env__.begin(write=True, buffers=True)
