#ifdef __INTELLISENSE__
#	include "pyscheduler/move_only.hpp"
#endif

#include <memory>

namespace pyscheduler {

template <typename R, typename... Args>
template <typename F>
MoveOnlyFunction<R(Args...)>::MoveOnlyFunction(F&& f)
	: _obj(new F(std::move(f)), [](void* p) { delete static_cast<F*>(p); })
	, _invoke([](void* p, Args... args) -> R {
		return (*static_cast<F*>(p))(std::forward<Args>(args)...);
	}) { }

template <typename R, typename... Args>
R MoveOnlyFunction<R(Args...)>::operator()(Args... args) {
	return _invoke(_obj.get(), std::forward<Args>(args)...);
}

template <typename R, typename... Args>
MoveOnlyFunction<R(Args...)>::operator bool() const {
	return (bool)_obj;
}
} // namespace pyscheduler