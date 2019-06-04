# [DL4AGX](https://github.com/NVIDIA/DL4AGX/compare/v1.1.0...master) (2019-06-03)

# [1.1.0](https://github.com/NVIDIA/DL4AGX/compare/v1.0.0...v1.1.0) (2019-06-03)

### Bug Fixes

* **//toolchain:** Default to PIC on all platforms ([2a1d7ec](https://github.com/NVIDIA/DL4AGX/commit/2a1d7ec))
* **//tools/linter:** Fix bazel not using python3 when using the linters ([cffdfca](https://github.com/NVIDIA/DL4AGX/commit/cffdfca))


* chore!: Freezing Bazel version at 0.26.0 ([d5bafa7](http://github.com/NVIDIA/DL4AGX/commit/d5bafa7))
* chore(//toolchains)!: Move cross compilation toolchains to new starlark based ones ([d0a65f3](http://github.com/NVIDIA/DL4AGX/commit/d0a65f3))
* Initial Open Sourcing of the Repository ([5fdf921](http://github.com/NVIDIA/DL4AGX/commit/5fdf921))


### BREAKING CHANGES

* Previous versions of bazel will no longer be supported. Bazel versions now will be bumped manually instead of using latest
* This is to support Bazel 0.25+ which deprecates CROSSTOOL. Please use bazel 0.25 or greater, there will be no more support on the CROSSTOOL toolchains

# [1.0.0](https://github.com/NVIDIA/DL4AGX/compare/5fdf9213...v1.0.0) (2019-05-01)


* Initial Open Sourcing of the Repository ([5fdf9213](https://github.com/NVIDIA/DL4AGX/commit/5fdf9213))
