#pragma once

// TIL (Apr 21, 2025) that need to specify symbol visibility or else the compiler can optimize functions away 😭😭😭😭
#if defined _WIN32 || defined __CYGWIN__
#	define PYSCHEDULER_LIBRARY_EXPORT __declspec(dllexport)
#	define PYSCHEDULER_LIBRARY_LOCAL
#else
#	define PYSCHEDULER_LIBRARY_EXPORT __attribute__((visibility("default")))
#	define PYSCHEDULER_LIBRARY_LOCAL __attribute__((visibility("hidden")))
#endif
