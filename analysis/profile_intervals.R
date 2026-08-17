# Profile-likelihood intervals for the confirmatory contrasts.
#
#   Rscript analysis/profile_intervals.R <frames-dir> [out-dir]
#
# <frames-dir> is the frames/ directory inside a run archive written by
# analysis/run.sh; out-dir defaults to it.
#
# WHY. The uptake models estimate the participant random-effect variance at
# exactly zero on this data - every grounding fit reports a singular
# convergence - and a Wald interval read off the curvature at a variance
# boundary is the weakest inference in the results. Profile likelihood does not
# assume the log-likelihood is quadratic there, so it is the interval to report
# for those contrasts. The count models are well away from any boundary and
# their Wald intervals are reported alongside as a check rather than replaced.
#
# WHAT IS FITTED. Two uptake specifications, because the manuscript and the
# platform disagree about one term and the difference is prespecified:
#   primary      stacked rows (unit x stage), stage fixed, participant intercept
#   unit_re      the same plus a unit intercept - the planned sensitivity, since
#                each unit contributes repeated stage observations
# Both use the frozen covariate set (scenario title, analytical position), and
# the count model is the same Poisson/negative-binomial rule as models.R.

suppressMessages({
  library(glmmTMB)
})

STAGES <- c("acknowledgement", "update_claim", "incorporation", "retention")
CONTRASTS <- list(
  list(id = "A", label = "vc_activation vs stable_converted",
       reference = "stable_converted", treated = "vc_activation"),
  list(id = "B", label = "vc_deactivation vs stable_natural",
       reference = "stable_natural", treated = "vc_deactivation")
)

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript profile_intervals.R <frames-dir> [out-dir]")
frames <- args[1]
out_dir <- if (length(args) > 1) args[2] else frames
units <- read.csv(file.path(frames, "unit_level.csv"), stringsAsFactors = FALSE)
scen <- read.csv(file.path(frames, "scenario_level.csv"), stringsAsFactors = FALSE)

# One row per observable unit-stage. A gated stage carries no value and is not
# an observed failure, so it is dropped rather than read as zero.
stack_units <- function(d) {
  out <- do.call(rbind, lapply(STAGES, function(s) {
    keep <- !is.na(d[[s]]) & trimws(as.character(d[[s]])) != ""
    if (!any(keep)) return(NULL)
    data.frame(d[keep, c("session_id", "participant_id", "condition",
                         "unit_index", "scenario_title", "analytical_position")],
               stage = s,
               grounding = as.integer(d[[s]][keep]),
               stringsAsFactors = FALSE)
  }))
  out$unit_key <- paste(out$session_id, out$unit_index, sep = "|")
  out$stage <- relevel(factor(out$stage), ref = "acknowledgement")
  out
}

stacked <- stack_units(units)

# Profile intervals fail loudly rather than silently falling back to Wald: a
# reported profile interval has to be one.
interval <- function(model, method) {
  ci <- tryCatch(
    confint(model, method = method, parm = "beta_"),
    error = function(e) NULL, warning = function(w) NULL)
  if (is.null(ci)) return(c(NA_real_, NA_real_))
  row <- grep("^cond", rownames(ci))
  if (!length(row)) return(c(NA_real_, NA_real_))
  as.numeric(ci[row[1], 1:2])
}

rows <- list()
for (cc in CONTRASTS) {
  keep <- c(cc$reference, cc$treated)

  for (spec in c("primary", "unit_re")) {
    sub <- stacked[stacked$condition %in% keep, ]
    sub$cond <- factor(sub$condition, levels = keep)
    form <- if (spec == "primary") {
      grounding ~ cond + stage + scenario_title + analytical_position +
        (1 | participant_id)
    } else {
      grounding ~ cond + stage + scenario_title + analytical_position +
        (1 | participant_id) + (1 | unit_key)
    }
    m <- glmmTMB(form, data = sub, family = binomial)
    est <- fixef(m)$cond[grep("^cond", names(fixef(m)$cond))][1]
    co <- summary(m)$coefficients$cond
    crow <- co[grep("^cond", rownames(co)), , drop = FALSE]
    prof <- interval(m, "profile")
    wald <- interval(m, "wald")
    rows[[length(rows) + 1]] <- data.frame(
      contrast = sprintf("%s: %s", cc$id, cc$label),
      outcome = "uptake (stacked)", spec = spec, model = "logistic GLMM",
      estimate_or = exp(as.numeric(crow[1, 1])),
      profile_low = exp(prof[1]), profile_high = exp(prof[2]),
      wald_low = exp(wald[1]), wald_high = exp(wald[2]),
      p = as.numeric(crow[1, 4]), n_obs = nrow(sub),
      n_participants = length(unique(sub$participant_id)),
      re_sd_participant = sqrt(as.numeric(VarCorr(m)$cond$participant_id)),
      stringsAsFactors = FALSE)
  }

  sub <- scen[scen$condition %in% keep, ]
  sub$cond <- factor(sub$condition, levels = keep)
  m <- glmmTMB(repair_post_boundary ~ cond + scenario_title +
                 analytical_position + (1 | participant_id),
               data = sub, family = poisson)
  ratio <- sum(residuals(m, type = "pearson")^2) / df.residual(m)
  label <- "Poisson GLMM"
  if (is.finite(ratio) && ratio > 1.5) {
    m <- update(m, family = nbinom2)
    label <- sprintf("NB GLMM (dispersion %.2f)", ratio)
  }
  co <- summary(m)$coefficients$cond
  crow <- co[grep("^cond", rownames(co)), , drop = FALSE]
  prof <- interval(m, "profile")
  wald <- interval(m, "wald")
  rows[[length(rows) + 1]] <- data.frame(
    contrast = sprintf("%s: %s", cc$id, cc$label),
    outcome = "post-transition repair", spec = "primary", model = label,
    estimate_or = exp(as.numeric(crow[1, 1])),
    profile_low = exp(prof[1]), profile_high = exp(prof[2]),
    wald_low = exp(wald[1]), wald_high = exp(wald[2]),
    p = as.numeric(crow[1, 4]), n_obs = nrow(sub),
    n_participants = length(unique(sub$participant_id)),
    re_sd_participant = sqrt(as.numeric(VarCorr(m)$cond$participant_id)),
    stringsAsFactors = FALSE)
}

res <- do.call(rbind, rows)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(out_dir, "profile_intervals.csv")
write.csv(res, out, row.names = FALSE)

cat("Ratios are odds ratios for uptake and incidence-rate ratios for repair.\n")
cat("profile_* is the likelihood interval; wald_* is the curvature interval.\n\n")
print(res[, c("contrast", "outcome", "spec", "estimate_or",
              "profile_low", "profile_high", "wald_low", "wald_high",
              "p", "n_obs", "re_sd_participant")], row.names = FALSE, digits = 3)
cat("\nwritten:", out, "\n")
