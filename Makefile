configure-debug:
	cmake -B build \
	-DCMAKE_BUILD_TYPE=Debug \
	.

configure-release:
	cmake -B build \
	-DCMAKE_BUILD_TYPE=Release \
	.

build:
	make -C build -j $(nproc)
.PHONY: build

.PHONY: configure
