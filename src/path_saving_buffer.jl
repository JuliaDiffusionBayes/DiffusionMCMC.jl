#===============================================================================

            Utility struct for saving most recently sampled paths.

===============================================================================#

mutable struct CyclicCounter
    first::Int64
    last::Int64
    i::Int64

    function CyclicCounter(n; first=1, start_from=first)
        new(first, n, start_from)
    end
end

function next!(cc::CyclicCounter)
    cc.i = mod1(cc.i+1, cc.last)
    cc.i < cc.first && (cc.i = cc.first)
end

(cc::CyclicCounter)() = cc.i
function (cc::CyclicCounter)(::Val{:next})
    i = cc.i
    next!(cc)
    i
end

struct PathSavingBuffer{TX}
    XX::TX
    iter::CyclicCounter

    function PathSavingBuffer(XX, N)
        XX_container = map(1:length(XX)) do i
            trajectory(
                glue_containers( map(x->x.t, XX[i]) ),
                glue_containers( map(x->x.x, XX[i]) ),
            )
        end
        XX_containers = [deepcopy(XX_container) for _ in 1:N]

        new{typeof(XX_containers)}(XX_containers, CyclicCounter(N))
    end
end

function glue_containers(xs)
    glued_xs = collect(Iterators.flatten(map(x->x[1:end-1], xs)))
    append!(glued_xs, [xs[end][end]])
    glued_xs
end

save!(buf::PathSavingBuffer, XX) = _copy_path!(buf.XX[buf.iter(_NEXT)], XX)

function _copy_path!(_to, _from)
    for i in eachindex(_to, _from)
        offset = 1
        for j in eachindex(_from[i])
            N = length(_from[i][j].x)
            view(_to[i].x, offset:offset+N-2) .= view(_from[i][j].x, 1:N-1)
            offset += N-1
        end
        _to[i].x[end] = _from[i][end].x[end]
    end
end
