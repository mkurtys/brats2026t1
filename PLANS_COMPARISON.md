# Plans Comparison: nnUNetPlans vs ResEncUNetLPlans

## Patch sizes (same spacing, same GPU budget)

| Plan | Patch size | Coverage of 240×240×155 | Batch |
|---|---|---|---|
| nnUNetPlans (PlainConvUNet) | [112, 160, 128] | ~73% | 2 |
| ResEncUNetLPlans (ResidualEncoderUNet) | [160, 224, 192] | ~96% | 2 |

Spacing is identical: [1.0, 0.898, 0.859] mm. Difference is purely architectural memory budget.

**Why bigger patch for ResEncUNet:** residual blocks in the encoder are more memory-efficient per feature map (skip connections enable thinner bottlenecks), so the same VRAM budget yields a larger patch. The "L" in ResEncUNetL was specifically designed for bigger patches.

## Implications for instance-uniform sampling

**At 96% coverage (ResEncUNetL):** patch centering barely matters for what's *visible* — all lesions are in the patch regardless of center. Instance-uniform sampling has minimal effect.

**What still matters at large patch sizes:**
- Augmentation quality: centering on a tiny lesion puts it near the center of the elastic deformation field (least distortion). Centering on a large region puts tiny lesions near the edge (more distortion, risk of cropping).
- This effect is small in practice at 96% coverage.

**What always matters regardless of patch size:**
- **Class-balanced case sampling** (weighting RC/NETC cases higher) — controls which *cases* appear in each batch, not where you crop within them. Keep this for ResEncUNetL.

**Conclusion:** For ResEncUNetL, keep class-balanced case sampling, but instance-uniform patch centering is doing little work. Instance-uniform sampling is more meaningful for the default plans (73% coverage).
