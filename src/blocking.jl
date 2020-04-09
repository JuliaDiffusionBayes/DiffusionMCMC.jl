"""
    BlockingChequerboardSwitch <: BlockingType

A decorator indicating a switch (in a chequerboard pattern) that changes the
locations of delimiters of blocks.
"""
struct BlockingChequerboardSwitch <: BlockingType
    block_half_len::Int64
end
