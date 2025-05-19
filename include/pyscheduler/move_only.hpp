#pragma once

#include <memory>

namespace pyscheduler {

template <typename Signature>
class MoveOnlyFunction;

template <typename R, typename... Args>
class MoveOnlyFunction<R(Args...)> {
public:
	MoveOnlyFunction() = default;

	template <typename F>
	MoveOnlyFunction(F&& f);

	MoveOnlyFunction(MoveOnlyFunction&&) noexcept = default;
	MoveOnlyFunction& operator=(MoveOnlyFunction&&) noexcept = default;

	MoveOnlyFunction(const MoveOnlyFunction&) = delete;
	MoveOnlyFunction& operator=(const MoveOnlyFunction&) = delete;

	R operator()(Args... args);
	explicit operator bool() const;

private:
	using InvokeFn = R (*)(void*, Args&&...);
	using DestroyFn = void (*)(void*);

	std::unique_ptr<void, DestroyFn> _obj{ nullptr, nullptr };
	InvokeFn _invoke{ nullptr };
};

} // namespace pyscheduler

#include "pyscheduler/details/move_only_impl.hpp"