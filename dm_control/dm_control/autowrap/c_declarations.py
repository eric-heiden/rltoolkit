# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Python representations of C declarations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

# Internal dependencies.

from dm_control.autowrap import codegen_util
from dm_control.autowrap import header_parsing

import six


class CDeclBase(object):
  """Base class for Python representations of C declarations."""

  def __init__(self, **attrs):
    self._attrs = attrs
    for k, v in six.iteritems(attrs):
      setattr(self, k, v)

  def __repr__(self):
    """Pretty string representation."""
    attr_str = ", ".join("{0}={1!r}".format(k, v)
                         for k, v in six.iteritems(self._attrs))
    return "{0}({1})".format(type(self).__name__, attr_str)

  @property
  def docstring(self):
    """Auto-generate a docstring for self."""
    return "\n".join(textwrap.wrap(self.comment, 74))

  @property
  def ctypes_typename(self):
    """ctypes typename."""
    return self.typename

  @property
  def ctypes_ptr(self):
    """String representation of self as a ctypes pointer."""
    return header_parsing.CTYPES_PTRS.get(
        self.ctypes_typename, "ctypes.POINTER({})".format(self.ctypes_typename))

  @property
  def np_dtype(self):
    """Get a numpy dtype name for self, fall back on self.ctypes_typename."""
    return header_parsing.CTYPES_TO_NUMPY.get(self.ctypes_typename,
                                              self.ctypes_typename)

  @property
  def np_flags(self):
    """Tuple of strings specifying numpy.ndarray flags."""
    return ("C", "W")


class Struct(CDeclBase):
  """C struct declaration."""

  def __init__(self, name, typename, members, sub_structs, comment="",
               parent=None, is_const=None):
    super(Struct, self).__init__(name=name,
                                 typename=typename,
                                 members=members,
                                 sub_structs=sub_structs,
                                 comment=comment,
                                 parent=parent,
                                 is_const=is_const)

  @property
  def ctypes_struct_decl(self):
    """Generates a ctypes.Structure declaration for self."""
    indent = codegen_util.Indenter()
    s = textwrap.dedent("""
    class {0.ctypes_typename:}(ctypes.Structure):
      \"\"\"{0.docstring:}\"\"\"
    """.format(self))
    with indent:
      if self.members:
        s += indent("\n_fields_ = [\n")
        with indent:
          with indent:
            s += ",\n".join(indent(m.ctypes_field_decl)
                            for m in six.itervalues(self.members))
        s += indent("\n]\n")
    return s

  @property
  def ctypes_typename(self):
    """Mangles ctypes.Structure typenames to distinguish them from wrappers."""
    return codegen_util.mangle_struct_typename(self.typename)

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    return "('{0.name:}', {0.ctypes_typename:})".format(self)   # pylint: disable=missing-format-attribute

  @property
  def wrapper_name(self):
    return codegen_util.camel_case(self.typename) + "Wrapper"

  @property
  def wrapper_class(self):
    """Generates a Python class containing getter/setter methods for members."""
    indent = codegen_util.Indenter()
    s = textwrap.dedent("""
    class {0.wrapper_name}(util.WrapperBase):
      \"\"\"{0.docstring:}\"\"\"
    """.format(self))
    with indent:
      s += "".join(indent(m.getters_setters)
                   for m in six.itervalues(self.members))
    return s

  @property
  def getters_setters(self):
    """Populates a Python class with getter & setter methods for self."""
    return textwrap.dedent("""
    @util.CachedProperty
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return {0.wrapper_name}(ctypes.pointer(self._ptr.contents.{0.name}))
    """.format(self))   # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """String representation of self as a ctypes function argument."""
    return self.ctypes_typename


class ScalarPrimitive(CDeclBase):
  """A scalar value corresponding to a C primitive type."""

  def __init__(self, name, typename, comment="", parent=None, is_const=None):
    super(ScalarPrimitive, self).__init__(name=name,
                                          typename=typename,
                                          comment=comment,
                                          parent=parent,
                                          is_const=is_const)

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    return "('{0.name:}', {0.ctypes_typename:})".format(self)   # pylint: disable=missing-format-attribute

  @property
  def getters_setters(self):
    """Populates a Python class with getter & setter methods for self."""
    return textwrap.dedent("""
    @property
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return self._ptr.contents.{0.name:}

    @{0.name:}.setter
    def {0.name:}(self, value):
      self._ptr.contents.{0.name:} = value
    """.format(self))   # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """String representation of self as a ctypes function argument."""
    return self.ctypes_typename


class ScalarPrimitivePtr(CDeclBase):
  """Pointer to a ScalarPrimitive."""

  def __init__(self, name, typename, comment="", parent=None, is_const=None):
    super(ScalarPrimitivePtr, self).__init__(name=name,
                                             typename=typename,
                                             comment=comment,
                                             parent=parent,
                                             is_const=is_const)

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    return "('{0.name:}', {0.ctypes_ptr:})".format(self)   # pylint: disable=missing-format-attribute

  @property
  def getters_setters(self):
    """Populates a Python class with getter & setter methods for self."""
    return textwrap.dedent("""
    @property
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return self._ptr.contents.{0.name:}

    @{0.name:}.setter
    def {0.name:}(self, value):
      self._ptr.contents.{0.name:} = value
    """.format(self))  # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """Generates string representation of self as a ctypes function argument."""
    # we assume that every pointer that maps to a numpy dtype corresponds to an
    # array argument/return value
    if self.ctypes_typename in header_parsing.CTYPES_TO_NUMPY:
      return ("util.ndptr(dtype={0.np_dtype}, flags={0.np_flags!s:})"
              "".format(self))  # pylint: disable=missing-format-attribute
    else:
      return self.ctypes_ptr


class StaticPtrArray(CDeclBase):
  """Array of arbitrary pointers whose size can be inferred from the headers."""

  def __init__(self, name, typename, shape, comment="", parent=None,
               is_const=None):
    super(StaticPtrArray, self).__init__(name=name,
                                         typename=typename,
                                         shape=shape,
                                         comment=comment,
                                         parent=parent,
                                         is_const=is_const)

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    if self.typename in header_parsing.CTYPES_PTRS:
      return "('{0.name:}', {0.ctypes_ptr:} * {1:})".format(  # pylint: disable=missing-format-attribute
          self, " * ".join(str(d) for d in self.shape))
    else:
      return "('{0.name:}', {0.ctypes_typename:} * {1:})".format(  # pylint: disable=missing-format-attribute
          self, " * ".join(str(d) for d in self.shape))

  @property
  def getters_setters(self):
    """Populates a Python class with getter & setter methods for self."""
    return textwrap.dedent("""
    @property
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return self._ptr.contents.{0.name:}
    """.format(self))  # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """Generates string representation of self as a ctypes function argument."""
    return "{0.ctypes_typename:}".format(self)


class StaticNDArray(CDeclBase):
  """Numeric array whose dimensions can all be inferred from the headers."""

  def __init__(self, name, typename, shape, comment="", parent=None,
               is_const=None):
    super(StaticNDArray, self).__init__(name=name,
                                        typename=typename,
                                        shape=shape,
                                        comment=comment,
                                        parent=parent,
                                        is_const=is_const)

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    return "('{0.name:}', {0.ctypes_typename:} * ({1:}))".format(  # pylint: disable=missing-format-attribute
        self, " * ".join(str(d) for d in self.shape))

  @property
  def getters_setters(self):
    """Populates a Python class with a getter method for self (no setter)."""
    return textwrap.dedent("""
    @util.CachedProperty
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return util.buf_to_npy(self._ptr.contents.{0.name:}, {0.shape!s:})
    """.format(self))  # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """Generates string representation of self as a ctypes function argument."""
    return ("util.ndptr(shape={0.shape}, dtype={0.np_dtype}, "  # pylint: disable=missing-format-attribute
            "flags={0.np_flags!s})".format(self))


class DynamicNDArray(CDeclBase):
  """Numeric array where one or more dimensions are determined at runtime."""

  def __init__(self, name, typename, shape, comment="", parent=None,
               is_const=None):
    super(DynamicNDArray, self).__init__(name=name,
                                         typename=typename,
                                         shape=shape,
                                         comment=comment,
                                         parent=parent,
                                         is_const=is_const)

  @property
  def runtime_shape_str(self):
    """String representation of shape tuple at runtime."""
    rs = []
    for d in self.shape:
      # dynamically-sized dimension
      if isinstance(d, str):
        if self.parent and d in self.parent.members:
          rs.append("self.{}".format(d))
        else:
          rs.append("self._model.{}".format(d))
      # static dimension
      else:
        rs.append(str(d))
    return str(tuple(rs)).replace("'", "")  # strip quotes from string rep

  @property
  def ctypes_field_decl(self):
    """Generates a declaration for self as a field of a ctypes.Structure."""
    return "('{0.name:}', {0.ctypes_ptr})".format(self)  # pylint: disable=missing-format-attribute

  @property
  def getters_setters(self):
    """Populates a Python class with a getter method for self (no setter)."""
    return textwrap.dedent("""
    @util.CachedProperty
    def {0.name:}(self):
      \"\"\"{0.docstring:}\"\"\"
      return util.buf_to_npy(self._ptr.contents.{0.name:},
                             {0.runtime_shape_str:})
    """.format(self))  # pylint: disable=missing-format-attribute

  @property
  def arg(self):
    """Generates string representation of self as a ctypes function argument."""
    return ("util.ndptr(dtype={0.np_dtype}, flags={0.np_flags!s:})"
            "".format(self))  # pylint: disable=missing-format-attribute


class Function(CDeclBase):
  """A function declaration including input type(s) and return type."""

  def __init__(self, name, arguments, return_value, comment=""):
    super(Function, self).__init__(name=name,
                                   arguments=arguments,
                                   return_value=return_value,
                                   comment=comment)

  def ctypes_func_decl(self, cdll_name):
    """Generates a ctypes function declaration."""
    indent = codegen_util.Indenter()
    # triple-quoted docstring
    s = ("{0:}.{1.name:}.__doc__ = \"\"\"\n{1.docstring:}\"\"\"\n"  # pylint: disable=missing-format-attribute
        ).format(cdll_name, self)
    # arguments
    s += "{0:}.{1.name:}.argtypes = [".format(cdll_name, self)  # pylint: disable=missing-format-attribute
    if len(self.arguments) > 1:
      s += "\n"
      with indent:
        with indent:
          s += ",\n".join(indent(a.arg) for a in six.itervalues(self.arguments))
      s += "\n"
    else:
      s += ", ".join(indent(a.arg) for a in six.itervalues(self.arguments))
    s += "]\n"
    # return value
    s += "{0:}.{1.name:}.restype = {2:}\n".format(  # pylint: disable=missing-format-attribute
        cdll_name, self, self.return_value.arg)
    return s

  @property
  def docstring(self):
    """Generates a docstring."""
    indent = codegen_util.Indenter()
    s = "\n".join(textwrap.wrap(self.comment, 80)) + "\n\nArgs:\n"
    with indent:
      for a in six.itervalues(self.arguments):
        s += indent("{a.name:}: {a.arg:}{const:}\n".format(
            a=a, const=(" <const>" if a.is_const else "")))
    s += "Returns:\n"
    with indent:
      s += indent("{0.return_value.arg}{1:}\n".format(  # pylint: disable=missing-format-attribute
          self, (" <const>" if self.return_value.is_const else "")))
    return s


class StaticStringArray(CDeclBase):
  """A string array of fixed dimensions exported by MuJoCo."""

  def __init__(self, name, shape, symbol_name):
    super(StaticStringArray, self).__init__(name=name,
                                            shape=shape,
                                            symbol_name=symbol_name)

  def ctypes_var_decl(self, cdll_name=""):
    """Generates a ctypes export statement."""

    ptr_str = "ctypes.c_char_p"
    for dim in self.shape[::-1]:
      ptr_str = "({0} * {1!s})".format(ptr_str, dim)

    return "{0} = {1}.in_dll({2}, {3!r})\n".format(
        self.name, ptr_str, cdll_name, self.symbol_name)


class FunctionPtr(CDeclBase):
  """A pointer to an externally defined C function."""

  def __init__(self, name, symbol_name, type_name=None):
    super(FunctionPtr, self).__init__(
        name=name, symbol_name=symbol_name, type_name=type_name)

  def ctypes_var_decl(self, cdll_name=""):
    """Generates a ctypes export statement."""

    return "ctypes.c_void_p.in_dll({0}, {1!r})".format(
        cdll_name, self.symbol_name)
