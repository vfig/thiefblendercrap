from typing import NewType, get_type_hints
import struct

int8 = NewType('int8', int)
int16 = NewType('int16', int)
int32 = NewType('int32', int)
uint8 = NewType('uint8', int)
uint16 = NewType('uint16', int)
uint32 = NewType('uint32', int)
float32 = NewType('float32', float)
bytes4 = NewType('bytes4', bytes)

struct_types = {
    int8: 'b',
    int16: 'h',
    int32: 'l',
    uint8: 'B',
    uint16: 'H',
    uint32: 'L',
    float32: 'f',
    bytes4: '4s',
    }

class Struct:
    def __init__(self, **kw):
        if self.__class__==Struct:
            raise TypeError("Cannot instantiate Struct itself, only subclasses")
        hints = get_type_hints(self.__class__)
        if len(hints)==0:
            raise TypeError(f"{self.__class__.__name__} has no fields defined")
        for name, typeref in hints.items():
            if name not in kw:
                raise KeyError(name)
            # default = typeref.__supertype__()
            # setattr(self, name, default)
        for name, value in kw.items():
            if name not in hints:
                raise KeyError(name)
            setattr(self, name, typeref(value))

    @classmethod
    def format_string(cls):
        fmt = ['@']
        hints = get_type_hints(cls)
        for name, typeref in hints.items():
            ch = struct_types[typeref]
            fmt.append(ch)
        return ''.join(fmt)

    @classmethod
    def read(cls, data):
        fmt = cls.format_string()
        hints = get_type_hints(cls)
        values = struct.unpack(fmt, data)
        args = dict(zip(hints.keys(), values))
        return cls(**args)

    def write(self):
        fmt = self.format_string()
        hints = get_type_hints(self)
        values = (getattr(self, name) for name in hints.keys())
        return struct.pack(fmt, *values)

class Foo(Struct):
    magic: bytes4
    radius: float32
    flags: uint32
    app_data: uint32

data = b'LGMM\xdb\x0f\x49\x40\x01\x00\x00\x00\xff\xff\xff\xff'
foo = Foo.read(data)
print(f"magic: {foo.magic}")
print(f"radius: {foo.radius:f}")
print(f"flags: {foo.flags:04x}")
print(f"app_data: {foo.app_data:04x}")
print()
data2 = foo.write()
print(f"data: {data2!r}")
print("      equal?", ("yes" if data==data2 else "no"))
