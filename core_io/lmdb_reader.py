import lmdb
import sys
import numpy as np

class LMDBModel:

    # Path to the LMDB
    lmdb_path = None

    # LMDB Environment handle
    __lmdb_env__ = None

    # LMDB context handle
    __lmdb_txn__ = None

    # LMDB Cursor for navigating data
    __lmdb_cursor__ = None

    ''' Constructor and De-constructor
    '''
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.__start_session__()

    def __del__(self):
        self.close_session()

    ''' Session Function
    '''
    def __start_session__(self):

        # Open LMDB file
        self.__lmdb_env__ = lmdb.open(self.lmdb_path, readonly=True)

        # Crete context
        self.__lmdb_txn__ = self.__lmdb_env__.begin(write=False)

        # Get the cursor of current lmdb
        self.__lmdb_cursor__ = self.__lmdb_txn__.cursor()

    def close_session(self):
        if self.__lmdb_env__ is not None:
            self.__lmdb_env__.close()
            self.__lmdb_env__ = None

    ''' Read Routines
    '''
    def read_by_key(self, key):

        """
        Read value in lmdb by providing the key
        :param key: the string that corresponding to the value
        :return: array data
        """
        value = self.__lmdb_cursor__.get(key.encode())
        return value

    def read_ndarray_by_key(self, key):
        value = self.__lmdb_cursor__.get(key.encode())
        return np.fromstring(value, dtype=np.float32)

    def len_entries(self):
        length = self.__lmdb_txn__.stat()['entries']
        return length

    ''' Static Utilities
    '''
    @staticmethod
    def convert_to_img(data):

        """
        Transpose the data from the Caffe's format to the normal format
        :param data: ndarray object with dimension of (3, h, w)
        :return: transposed ndarray with dimension of (h, w, 3)
        """
        return data.transpose((1, 2, 0))
