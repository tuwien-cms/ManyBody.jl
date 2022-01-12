export
    Term,
    creator,
    annihilator,
    occupation,
    vacancy

"""
Represent product of fermionic creation/annihilation operators.

For a Fock space of flavours, this is an abstract representation of the product
of an arbitrary number of creation and annihilation operators, not necessarily
normal ordered.  One can write each such product in a canonical form:

    t(v,i,j,k) = v * prod(x -> c[x]', i) * prod(x -> c[x], reverse(j)) *
                 prod(x -> c[x]*c[x]', k)

where `i`, `j` are tuples of flavours, each one ordered ascendingly, and `k`
is a tuple of flavours disjoint from both `i` and `j`.  Note that above
expression is not a normal-ordered product either, but a "classification" of
each flavour with respect to the term's effect.

Instead of the `O(2^(2n))` dense or `O(2^n)` sparse storage required for the
elements of such a product, this class stores only an `O(1)` set of bitfields
allowing for fast on-the-fly computation.
"""
struct Term{T}
    mask::UInt64
    left::UInt64
    right::UInt64
    change::UInt64
    sign_mask::UInt64
    value::T
end

"""Conversion between terms types"""
Term{T}(term::Term{U}) where {T, U} = Term(term, U(term.value))

"""Convert term to itself"""
Term{T}(term::Term{T}) where {T} = term

"""Create a term from template, but assign new value to it"""
Term(term::Term, new_value) =
    Term(term.mask, term.left, term.right, term.change, term.sign_mask, new_value)

"""Constant term (energy shift)"""
Term(value::Number) =
    Term(UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), value)

"""Make creation operator"""
function creator(i::Integer; value=Int8(1), statmask::UInt64=~UInt64(0))
    @boundscheck (i >= 1 && i <= 64) || throw(BoundsError(1:64, i))
    mask = UInt64(1) << (i - 1)
    signmask = (mask - one(mask)) & statmask
    return Term(mask, mask, zero(mask), mask, signmask, value)
end

"""Make annihilation operator"""
function annihilator(i::Integer; value=Int8(1), statmask::UInt64=~UInt64(0))
    @boundscheck (i >= 1 && i <= 64) || throw(BoundsError(1:64, i))
    mask = UInt64(1) << (i - 1)
    signmask = (mask - one(mask)) & statmask
    return Term(mask, zero(mask), mask, mask, signmask, value)
end

"""Make occupation (number) operator"""
function occupation(i::Integer; value=Int8(1))
    @boundscheck (i >= 1 && i <= 64) || throw(BoundsError(1:64, i))
    mask = UInt64(1) << (i - 1)
    return Term(mask, mask, mask, zero(mask), zero(mask), value)
end

"""Make vacancy (1 - number) operator"""
function vacancy(i::Integer; value=Int8(1))
    @boundscheck (i >= 1 && i <= 64) || throw(BoundsError(1:64, i))
    mask = UInt64(1) << (i - 1)
    return Term(mask, zero(mask), zero(mask), zero(mask), zero(mask), value)
end

"""Multiplies two terms"""
function Base.:*(a::Term, b::Term)
    # Get value first, mainly to determine the type of result and ensure
    # type stability.  We have to make sure to get a signed type here, as
    # the sign may flip later (this is handled in a specialization).
    value = a.value * b.value

    # First, take care of the Pauli principle.  This is relatively easy, as
    # a.right and b.left are the "demands" of the two terms on their in-between
    # state, each confined to their mask.  This means we simply have to fulfill
    # both demands whenever the masks intersect.
    intersect = a.mask & b.mask
    if xor(a.right, b.left) & intersect != 0
        return Term(zero(value))
    end

    # Now that we know that the term is valid, let us determine the left and
    # right outer states: a.left and b.right are again demands on those states,
    # each confined to their mask.  Outside the mask, the other term may make
    # demands "through" the adjacent term.
    mask = a.mask | b.mask
    right = b.right | (a.right & ~b.mask)
    left = a.left | (b.left & ~a.mask)
    change = xor(left, right)

    # Each fundamental operator comes with a sign mask extending to the less
    # significant side of it.  Ignoring the flavours affected by the term for
    # the moment, multiplying terms simply means xor'ing the masks.
    sign_mask = xor(a.sign_mask, b.sign_mask)

    # Note that here, the order matters: for a normal-ordered term, we assume
    # that:
    #
    #       ψ_L = c[i[1]]' * ... * c[i[n]]' * c[j[n]] * ... * c[j[1]] ψ_R
    #
    # where both i and j are ordered ascendingly. This is because we can then
    # split this equation into two parts:
    #
    #       ψ_L = c[i[1]]' * ... * c[i[n]]' ψ_0
    #       ψ_R = c[j[1]]' * ... * c[j[n]]' ψ_0
    #
    # This means that for determining the sign, the intermediate state ψ_0 is
    # relevant, and so for a sign computation we must exclude any flavour in
    # ψ_L and ψ_R affected by the creators/annihilators.
    sign_mask &= ~mask

    # Finally we have to restore normal ordering by permuting operators.  We
    # know this may only yield a global sign.  So, we simply observe the
    # difference in effect of the product and normal-ordered product on a
    # single contributing "trial" state.  We choose the trial to be `right`,
    # as we know the normal-ordered product contributes and does not produce
    # a sign.
    trial = right
    permute = trial & b.sign_mask
    trial = xor(trial, b.change)
    permute = xor(permute, trial & a.sign_mask)
    value = copy_parity(permute, value)

    return Term(mask, left, right, change, sign_mask, value)
end

# This ensures type-stability is observed.
Base.:*(a::Term{A}, b::Term{B}) where {A <: Unsigned, B <: Unsigned} =
    Term(a, signed(a.value)) * Term(b, signed(b.value))

"""Check if state is mapped when multiplied by term from the left"""
is_mapped_right(t::Term, state) = state & t.mask == t.right

"""Check if state is mapped when multiplied by term from the right"""
is_mapped_left(t::Term, state) = state & t.mask == t.left

"""Change sign of number if parity of a bit pattern is odd"""
copy_parity(pattern::Integer, number) =
    (count_ones(pattern) & 1 != 0) ? -number : number

# We use this to ensure type stability: otherwise, passing an unsigned
# number would have to return a union, as -x converts it to a signed one.
# We could silently use signed() here, but it is better to handle this
# explicitly in application code.
copy_parity(::Integer, ::Unsigned) =
    throw(DomainError("Cannot copy parity sign to unsigned number"))

"""Map a state through term, assuming that it is indeed mapped"""
map_state_right(t::Term, state) =
    xor(state, t.change), copy_parity(state & t.sign_mask, t.value)

"""Map a state through term, assuming that it is indeed mapped"""
map_state_left(t::Term, state) = map_state_right(t, state)

"""
Return the state shift through the term.

Index of the side diagonal where the values of term are nonzero, or,
equivalently, a state with index `i` is mapped by `term` to either nothing
or a state with index `i + state_shift(term)`.
"""
state_shift(t::Term) = (t.left - t.right) % Int64

"""Check if terms can be added and give one term"""
addable(a::Term, b::Term) =
    a.mask == b.mask && a.right == b.right && a.left == b.left

"""Convert terms with values of one type to another"""
Base.convert(::Type{Term{T}}, term::Term) where T =
    Term(term, convert(T, term.value))

"""Promotion rules"""
Base.promote_rule(::Type{Term{T}}, ::Type{Term{U}}) where {T, U} =
    Term{promote_type(T, U)}

"""Promotion rules"""
Base.promote_rule(::Type{Term{T}}, ::Type{Term{U}}) where {T <: Unsigned, U <: Unsigned} =
    Term{promote_type(signed(T), signed(U))}

"""Explicit plus of term"""
Base.:+(term::Term) = Term(term, +term.value)

"""Negate term"""
Base.:-(term::Term) = Term(term, -term.value)

"""Scale term by a scalar factor"""
Base.:*(factor::Number, term::Term) = Term(term, factor * term.value)

"""Scale term by a scalar factor"""
Base.:*(term::Term, factor::Number) = Term(term, term.value * factor)

"""Add terms with the same operators"""
Base.:+(a::Term, b::Term) = Term(_check_addable(a, b), a.value + b.value)

"""Subtract terms with the same operators"""
Base.:-(a::Term, b::Term) = Term(_check_addable(a, b), a.value - b.value)

"""Check if terms are equal"""
Base.:(==)(lhs::Term, rhs::Term) = addable(lhs, rhs) && lhs.value == rhs.value

"""Check if terms are approximately equal"""
function Base.isapprox(lhs::Term, rhs::Term; atol::Real=0,
                       rtol::Real=Base.rtoldefault(lhs.value, rhs.value, atol),
                       nans::Bool=false)
    return addable(lhs, rhs) && isapprox(lhs.value, rhs.value; atol, rtol, nans)
end

"""Get the transpose of a term"""
Base.transpose(t::Term) =
    Term(t.mask, t.right, t.left, t.change, t.sign_mask, t.value)

"""Get the adjoint (conjugate transpose) of a term"""
Base.adjoint(t::Term) =
    Term(t.mask, t.right, t.left, t.change, t.sign_mask, conj(t.value))

"""Get term like template, but with zero value"""
Base.zero(t::Term) = Term(t, zero(t.value))

"""Check if the value of term is zero"""
Base.iszero(t::Term) = iszero(t.value)

function _check_addable(a::Term, b::Term)
    if !addable(a, b)
        throw(ArgumentError("can only add terms which have same operators"))
    end
    return a
end
