# Paper Validating conditional density models and bayesian inference algorithms

## EXISTING DIAGNOSTIC ARE INTENSITIVE TO COVARIATE TRANSFORMATION

* Def 1: Global cosistancy:
  * An estimate $\hat{f}(y\mid x)$ is globally constency with the density $f(y\mid x)$  when $ H_{0}: \widehat{f}(y \mid \mathbf{x})=f(y \mid \mathbf{x})$ for almost every $x$ and $y$

Many methods callibrate density models by computing **PIT** values

* Def 2:  **PIT** Fix $\textbf{x}$ and $y$, probability integral transform of y at $\textbf{x}$.
  *  $P I T(y ; \mathbf{x})=\int_{-\infty}^{y} \widehat{f}\left(y^{\prime} \mid \mathbf{x}\right) d y^{\prime}$
  *  if $\hat{f}(y\mid x )$ is global consitency, $
\operatorname{PIT}\left(Y_{1} ; \mathbf{X}_{1}\right), \ldots, \operatorname{PIT}\left(Y_{n} ; \mathbf{X}_{n}\right) \stackrel{i . i . d}{\sim} \operatorname{Unif}(0,1)
$

* Def 3: Local consistancy: 
$
H_{0}(\mathbf{x}): \widehat{f}(y \mid \mathbf{x})=f(y \mid \mathbf{x}) \text { for every } y \in \mathscr{Y}$
  * A
  * B

## New diagnostics test local and global consitency

### **Theorem 2** (Local consitency and pointwise uniformity). Here, we combine local cositency definition with PIT.$H_{0}(\mathbf{x}): \widehat{f}(y \mid \mathbf{x})=f(y \mid \mathbf{x}) \text { for every } y \in \mathscr{Y}$ if and only if $PIT(Y,\mathbf{X}) \sim U(0,1)$, given X.

### However, data $(X,Y)$ atmost just appear one time at specific X. Therefore, it is hard to check this condition. (Emperically calculate $PIT(Y,X)$ need integral Y) .

* Local and global coverage test

* Amortized local p-p plots

* handling multi variate responses
