## LaSA upstream

The optional integration patch targets the public LaSA implementation by
Haoyu Ji and coauthors:

- Repository: <https://github.com/HaoyuJi/LaSA>
- Verified source revision: `f823dc8354e90251a0bd2f2b89fbd14eb4a9d047`
- Source file: [`libs/postprocess.py` at that revision](https://github.com/HaoyuJi/LaSA/blob/f823dc8354e90251a0bd2f2b89fbd14eb4a9d047/libs/postprocess.py)
- Upstream license: MIT, copyright (c) 2024 Haoyu Ji; see the upstream
  [`LICENSE`](https://github.com/HaoyuJi/LaSA/blob/f823dc8354e90251a0bd2f2b89fbd14eb4a9d047/LICENSE)
  and the retained local text at [`third_party/LaSA_LICENSE`](third_party/LaSA_LICENSE).

This demo does not redistribute upstream LaSA source files. The integration
patch is a small set of changes against the source file above. If applying it,
retain the upstream copyright and license notices in the LaSA repository.
