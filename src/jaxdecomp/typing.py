from collections.abc import Sequence

GdimsType = tuple[int, int, int]
# pdims is only two integers
# but in some cases we need to represent ('x' , None , 'y') as (Nx , 1 , Ny)
TransposablePdimsType = tuple[int, int, int]
PdimsType = tuple[int, int]

HaloExtentType = tuple[int, int]
Periodicity = tuple[bool, bool]

HLOResult = Sequence[type]
