# Scripts

Code that produces content of the site: the figures of the blog posts, and anything else generated
rather than written by hand. Each script writes its output directly into `agent/public/`, so a deleted or
outdated image can always be rebuilt from here.

| Path | Produces | Used in |
| --- | --- | --- |
| `generate-parameter-norm-curves.py` | `public/images/blog/grokking/parameter-norm-reparameterization.svg` | `src/content/blog/grokking-beyond-l2.md` |
| `eigenpairs-from-matrix-vector-products/` | every figure in `public/images/blog/matrix-vector-products/` (see its README) | `src/content/blog/eigenpairs-from-matrix-vector-products.md` |

Code for a new post goes in a folder named after the post, `agent/scripts/<post-slug>/`, with a README
saying how to run it, which files it writes, and the software versions used for the published output.
