from typing import Tuple, Type, Sequence

GdimsType = Tuple[int, int, int]
# pdims is only two integers
# but in some cases we need to represent ('x' , None , 'y') as (Nx , 1 , Ny)
TransposablePdimsType = Tuple[int, int, int]
PdimsType = Tuple[int, int]

HaloExtentType = Tuple[int, int]
Periodicity = Tuple[bool, bool]

HLOResult = Sequence[Type]
