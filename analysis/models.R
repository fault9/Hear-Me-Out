# Confirmatory models per the method chapter's Analysis section.
#
#   Rscript models.R --permute        pipeline smoke test (labels shuffled
#                                     within participant; results meaningless)
#   Rscript models.R --confirmatory   the once-only real run
#
# Two primary contrasts, each holding the post-transition voice state constant:
#   A: vc_activation   vs stable_converted
#   B: vc_deactivation vs stable_natural
# Co-primary outcomes: demonstrated_grounding (logistic GLMM) and
# repair_post_boundary (Poisson GLMM, negative-binomial if overdispersed).
# The four tests form one Holm-adjusted family.

suppressMessages({
  library(lme4)
  library(glmmTMB)
})

args <- commandArgs(trailingOnly = TRUE)
permute <- "--permute" %in% args
confirmatory <- "--confirmatory" %in% args
if (!xor(permute, confirmatory)) {
  stop("pass exactly one of --permute or --confirmatory")
}

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
frames <- file.path(here, "output", "frames")
dat <- read.csv(file.path(frames, "scenario_level.csv"), stringsAsFactors = FALSE)

dat$grounding <- as.integer(dat$demonstrated_grounding %in% c("1", "true", "True", "TRUE"))
dat$repairs_post <- suppressWarnings(as.integer(dat$repair_post_boundary))
dat$participant_id <- factor(dat$participant_id)
dat$scenario_title <- factor(dat$scenario_title)
dat$position <- suppressWarnings(as.integer(dat$analytical_position))

if (permute) {
  cat("=== PERMUTED SMOKE RUN — condition labels shuffled; results are meaningless ===\n")
  set.seed(20260805)
  dat$condition <- ave(dat$condition, dat$participant_id,
                       FUN = function(x) sample(x))
} else {
  cat("=== CONFIRMATORY RUN — this is the once-only real analysis ===\n")
}

contrasts <- list(
  A = c(treated = "vc_activation", reference = "stable_converted"),
  B = c(treated = "vc_deactivation", reference = "stable_natural")
)

fit_one <- function(sub, outcome) {
  sub$cond <- factor(sub$condition,
                     levels = c(sub$reference[1], sub$treated[1]))
  if (outcome == "grounding") {
    m <- glmer(grounding ~ cond + scenario_title + position + (1 | participant_id),
               data = sub, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
    co <- summary(m)$coefficients
    row <- co[grep("^cond", rownames(co)), , drop = FALSE]
    return(list(estimate = row[1, 1], se = row[1, 2], p = row[1, 4],
                model = "logistic GLMM", n = nrow(sub)))
  }
  # Repairs: Poisson first; refit negative-binomial when overdispersed
  # (Pearson chi-square / df > 1.5).
  m <- glmmTMB(repairs_post ~ cond + scenario_title + position + (1 | participant_id),
               data = sub, family = poisson)
  pearson <- sum(residuals(m, type = "pearson")^2)
  ratio <- pearson / df.residual(m)
  label <- "Poisson GLMM"
  if (is.finite(ratio) && ratio > 1.5) {
    m <- update(m, family = nbinom2)
    label <- sprintf("NB GLMM (dispersion %.2f)", ratio)
  }
  co <- summary(m)$coefficients$cond
  row <- co[grep("^cond", rownames(co)), , drop = FALSE]
  list(estimate = row[1, 1], se = row[1, 2], p = row[1, 4],
       model = label, n = nrow(sub))
}

results <- data.frame()
for (name in names(contrasts)) {
  pair <- contrasts[[name]]
  sub <- dat[dat$condition %in% pair, ]
  sub$treated <- pair[["treated"]]
  sub$reference <- pair[["reference"]]
  sub <- sub[!is.na(sub$position), ]
  for (outcome in c("grounding", "repairs_post")) {
    keep <- if (outcome == "repairs_post") !is.na(sub$repairs_post) else rep(TRUE, nrow(sub))
    fit <- tryCatch(fit_one(sub[keep, ], outcome),
                    error = function(e) list(estimate = NA, se = NA, p = NA,
                                             model = paste("FAILED:", conditionMessage(e)),
                                             n = sum(keep)))
    results <- rbind(results, data.frame(
      contrast = sprintf("%s: %s vs %s", name, pair[["treated"]], pair[["reference"]]),
      outcome = outcome, model = fit$model, estimate = fit$estimate,
      se = fit$se, p = fit$p, n_obs = fit$n,
      n_participants = length(unique(sub$participant_id)),
      permuted = permute))
  }
}

results$p_holm <- p.adjust(results$p, method = "holm")
out <- file.path(here, "output",
                 if (permute) "smoke_results.csv" else "confirmatory_results.csv")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
write.csv(results, out, row.names = FALSE)
print(results[, c("contrast", "outcome", "model", "estimate", "p", "p_holm", "n_obs")])
cat("\nwritten:", out, "\n")
