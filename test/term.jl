using Test
using ManyBody
using LinearAlgebra

@testset "Term" begin

@testset "pauli" begin
    @test iszero(zero(Term(1.0)))
    @test iszero(creator(3) * creator(3))
    @test iszero(creator(1) * annihilator(4) * annihilator(4))
end

@testset "convert" begin
    @test Term(2.0) == Term(2)
    @test creator(3) == Term{Int64}(creator(3))
end

@testset "normal ordering" begin
    c = annihilator
    cdag = creator

    t = cdag(7) * cdag(4) * c(2) * c(6)
    @test -t == cdag(4) * cdag(7) * c(2) * c(6)
    @test t == cdag(4) * cdag(7) * c(6) * c(2)

    t2 = c(9) * c(5) * c(3)
    @test t * t2 == +cdag(4) * cdag(7) * c(9) * c(6) * c(5)  * c(3) * c(2)

    # Sign not sure
    @test t2 * t == +cdag(4) * cdag(7) * c(9) * c(6) * c(5)  * c(3) * c(2)
end

@testset "vacancy" begin
    c = annihilator
    cdag = creator
    n = occupation
    v = vacancy

    t1 = 10 * n(2)
    @test t1 * t1 == 10 * t1

    t1 = 4 * v(4)
    @test t1 * t1 == 4 * t1

    t1 = 3 * c(3) * v(2) * cdag(1)
    @test t1 * v(2) == t1
end

@testset "quadraticterm" begin
    t = 2.0 * creator(1) * annihilator(3)
    tprime = 2.0 * creator(3) * annihilator(1)

    @test t' == tprime
    @test transpose(t) == tprime
end

end
