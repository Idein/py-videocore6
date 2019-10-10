
from cpython.buffer cimport Py_buffer, PyBUF_SIMPLE, \
        PyObject_GetBuffer, PyBuffer_Release


def get_ptr(object mem):

    cdef Py_buffer buf

    cdef int err = PyObject_GetBuffer(mem, &buf, PyBUF_SIMPLE)
    if err:
        raise RuntimeError(f'Failed to get buffer from object: {err}')

    cdef unsigned long ptr = <unsigned long> buf.buf

    PyBuffer_Release(&buf)

    return ptr
