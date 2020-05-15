struct ProcessorNumberError <: Exception 
	p :: Int
	np :: Int
end
function Base.showerror(io::IO,err::ProcessorNumberError)
	print(io,"processor id $(err.p) does not line in the range $(1:err.np)")
end

struct DecreasingIteratorError <: Exception 
end
function Base.showerror(io::IO,err::DecreasingIteratorError)
	print(io,"all the iterators need to be strictly increasing")
end

struct TaskNotPresentError{T,U} <: Exception
	t :: T
	task :: U
end
function Base.showerror(io::IO,err::TaskNotPresentError)
	print(io,"could not find the task $(err.task) in the list $(err.t)")
end