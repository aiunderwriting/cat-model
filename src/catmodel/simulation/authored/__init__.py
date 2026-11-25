"""Loss terms only hand-authored setups use; import each from its module.

``setup build`` freezes authored laws, which may carry them; a setup imported
from collected answers never does: its episodes have exactly one member and
its laws carry no payment kernel. ``payments`` reads a payment kernel's paid
fractions. The kernel's developer-liability share is sampled by
``catmodel.probability.liability``. The engine reaches these terms only
through a law that names them.
"""
