struct TaskNotPresentError{T,U} <: Exception
	t :: T
	task :: U
end
function Base.showerror(io::IO,err::TaskNotPresentError)
	print(io,"could not find the task $(err.task) in the list $(err.t)")
end