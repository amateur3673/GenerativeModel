# GAN-VAE loss

For image $x$, feed to encoder, we can get:

$$
\mu_z,\sigma_z = Enc(x)
$$

New $z$ can be formed by: $Z=\mu+\epsilon\sigma_z$, $\epsilon$ is randomly sampled in $N(0,I)$.

Decode $Z$: $\tilde{X}=Dec(Z)$

Feeding both $X,Z$ to Dis to get $Dis_l(X), Dis_l(Z)$

Subsample: $Z_p\sim N(0,I)$, decode $X_p=Dec(Z_p)$

Loss:

KL loss: $KL(q(Z\mid X)\|p(Z))$

Disl loss: $-E_{q(Z\mid X)}[p(Dis_l(X)\mid Z)]$

GAN loss: $log(Dis(X))+log(1-Dis(\tilde{X}))+log(1-Dis(X_p))$