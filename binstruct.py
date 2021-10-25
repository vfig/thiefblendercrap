__all__ = (
    'int8',
    'int16',
    'int32',
    'uint8',
    'uint16',
    'uint32',
    'float32',
    'ByteString',
    'Array',
    'Struct',
    'StructView',
    )

import struct
from typing import get_type_hints

class PrimitiveTypeMixin:
    _format_string = ''
    _name = ''
    @classmethod
    def size(cls):
        return struct.calcsize(cls._format_string)
    @classmethod
    def read(cls, view, offset=0):
        values = struct.unpack_from(cls._format_string, view, offset=offset)
        return cls(values[0])
    def write(self, f):
        f.write(struct.pack(self.__class__._format_string, self))
    def __repr__(self):
        return f"{self._name}({self})"

def make_primitive_type(name_, base_type, format_string):
    class _PrimitiveType(PrimitiveTypeMixin, base_type):
        _format_string = format_string
        _name = name_
    _PrimitiveType.__name__ = name_
    return _PrimitiveType

int8 = make_primitive_type('int8', int, 'b')
int16 = make_primitive_type('int16', int, 'h')
int32 = make_primitive_type('int32', int, 'l')
uint8 = make_primitive_type('uint8', int, 'B')
uint16 = make_primitive_type('uint16', int, 'H')
uint32 = make_primitive_type('uint32', int, 'L')
float32 = make_primitive_type('float32', float, 'f')

def ByteString(length):
    return make_primitive_type(f'ByteString({length})', bytes, f'{length}s')

class ArrayInstance:
    typeref = None
    count = 0
    def __init__(self, values):
        if len(values) != self.count:
            raise ValueError(f"Expected {self.count} values, got {values!r}")
        self.values = [self.typeref(v) for v in values]
    def __getitem__(self, i):
        return self.values[i]
    def __len__(self):
        return self.count
    def __repr__(self):
        return f"<ArrayInstance of {self.count} x {self.typeref.__name__}>"

def Array(typeref_, count_):
    class TypedArrayInstance(ArrayInstance):
        typeref = typeref_
        count = count_
        @classmethod
        def size(cls):
            return cls.count*cls.typeref.size()
        @classmethod
        def read(cls, view, offset=0):
            values = []
            stride = cls.typeref.size()
            for i in range(cls.count):
                value = cls.typeref.read(view, offset=offset)
                offset += stride
                values.append(value)
            return cls(values)
        def write(self, f):
            for i in range(self.count):
                self.values[i].write(f)
    TypedArrayInstance.__name__ = f"{typeref_.__name__}x{count_}"
    return TypedArrayInstance

# TODO: can i turn Struct into a decorate that uses @dataclass and adds the
# read/write/size methods??
# TODO: also rename this because it clashes with struct.Struct haha
class Struct:
    def __init__(self, values=None):
        if self.__class__==Struct:
            raise TypeError("Cannot instantiate Struct itself, only subclasses")
        hints = get_type_hints(self.__class__)
        if len(hints)==0:
            raise TypeError(f"{self.__class__.__name__} has no fields defined")
        if values is None:
            values = self.default_values()
        if len(values)!=len(hints):
            raise ValueError(f"Expected {len(self.hints)} values")
        for (name, typeref), value in zip(hints.items(), values):
            setattr(self, name, typeref(value))

    @classmethod
    def size(cls):
        size = 0
        # TODO: this ignores padding and alignment!
        hints = get_type_hints(cls)
        for name, typeref in hints.items():
            size += typeref.size()
        return size

    @classmethod
    def read(cls, data, offset=0):
        hints = get_type_hints(cls)
        values = []
        for name, typeref in hints.items():
            value = typeref.read(data, offset=offset)
            offset += typeref.size()
            values.append(value)
        return cls(values)

    def write(self, f):
        hints = get_type_hints(self.__class__)
        for name, typeref in hints.items():
            value = getattr(self, name)
            value.write(f)

class StructView:
    def __init__(self, view, typeref, *, offset=0, count=-1, size=-1):
        self.view = view
        self.typeref = typeref
        self.stride = typeref.size()
        self.offset = offset
        if count==-1 and size==-1:
            raise ValueError("Must provide either count, or size, or neither (to use entire view)")
        if count!=-1:
            self.count = count
        elif size!=-1:
            self.count = size//self.stride
        else:
            self.count = len(view)//self.stride

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.step is not None and i.step != 1:
                raise ValueError("Slices with step size other than 1 are not supported.")
            if i.start >= 0:
                start = min(max(0, i.start), self.count-1)
            else:
                start = min(max(0, self.count+i.start), self.count-1)
            offset = self.offset+start*self.stride
            if i.stop >= 0:
                count = min(self.count, i.stop-i.start)
            else:
                count = min(self.count, self.count+i.stop-i.start)
            return self.__class__(self.view, self.typeref,
                offset=offset, count=count)
        else:
            if not (0<=i<self.count):
                raise IndexError(i)
            offset = self.offset+i*self.stride
            return self.typeref.read(self.view, offset=offset)

    def size(self):
        return self.count*self.stride
