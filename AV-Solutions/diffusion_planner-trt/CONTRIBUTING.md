# Contributing to This Project

We welcome community contributions including bug fixes, feature enhancements, documentation improvements, and issue reports.

Please review the following guidelines before submitting your contribution.

---

## Development Guidelines

1. **Fork this repository** and create your branch from `main` or the appropriate development branch.
2. **Make your changes**, ensuring code is clean, readable, and follows existing style patterns.
3. **Test your code** to verify that it works as expected and doesn't break existing functionality.
4. **Sign off your commits** using the `-s` flag (see below).
5. **Open a pull request** with a clear description of the issue being addressed or the feature being added.

---

## Developer Certificate of Origin (DCO)

All contributions must comply with the Developer Certificate of Origin (DCO) version 1.1.

By contributing, you certify the following:

```bash
Developer Certificate of Origin
Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
---

## Sign-Off Instructions

Each commit must be signed off to indicate agreement with the DCO. Use the `-s` flag when committing:

```bash
git commit -s -m "Your commit message"
```
This will append a `Signed-off-by`: line to your commit message.

If you've already made commits without sign-off, you can fix them with:
```bash
git rebase --signoff HEAD~<number-of-commits>
```

---
## Licensing
All contributions to this project will be licensed under the Apache License 2.0.

By submitting a pull request, you agree to license your contribution under the project's license.
