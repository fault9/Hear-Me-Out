# The four confirmatory tests in glmmTMB and ordinal, as the reported fit.
#
#   Rscript analysis/confirmatory_r.R <frames-dir> [out-dir]
#
# <frames-dir> is the frames/ directory inside a run archive from
# analysis/run.sh. Writes confirmatory_r.csv (components and hypotheses) next
# to it, or to out-dir.
#
# This reproduces the frozen analysis plan in established packages so the
# manuscript can report standard software rather than a purpose-written
# estimator. The specification is the plan's, transcribed from
# voice-paper/analysis/confirmatory.py so the two are the same model and not
# merely the same intention:
#
#   repair    repair_post_boundary ~ condition + scenario + position
#             + (1 | participant); Poisson unless the prespecified LRT for
#             alpha = 0 (50:50 chi2(0)/chi2(1) boundary mixture) rejects at .05
#   uptake    attained ~ stage + condition + position + unit_identity
#             + (1 | participant) + (1 | interaction), over unit x stage rows.
#             The interaction intercept is not optional: condition varies at the
#             interaction level, so without it the stage rows within a session
#             count as independent and the condition standard error is too small
#   H2a       the uptake model with the seven-point rating as a covariate
#   H2b       cumulative-link mixed model, rating ~ repair_total + condition
#             + scenario + position + (1 | participant)
#
# Reference levels are the plan's: stable_natural, acknowledgement, position 1.
# Each hypothesis pairs two components and is intersection-union: its p is the
# LARGER of the two, so both have to move. Holm adjusts across the four.

suppressMessages({
  library(glmmTMB)
  library(ordinal)
})

CONDITION_REF <- "stable_natural"
STAGE_REF <- "acknowledgement"
POSITION_REF <- "1"
STAGES <- c("acknowledgement", "update_claim", "incorporation", "retention")

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("usage: Rscript confirmatory_r.R <frames-dir> [out-dir]")
frames <- args[1]
out_dir <- if (length(args) > 1) args[2] else frames

units <- read.csv(file.path(frames, "unit_level.csv"), stringsAsFactors = FALSE)
scen <- read.csv(file.path(frames, "scenario_level.csv"), stringsAsFactors = FALSE)

prep <- function(d, ref) {
  d$condition <- relevel(factor(d$condition), ref = ref)
  d$scenario_title <- factor(d$scenario_title)
  d$analytical_position <- relevel(factor(as.character(d$analytical_position)),
                                   ref = POSITION_REF)
  d
}

# Unit x stage rows. A gated stage carries no value and is unobservable rather
# than a failure, so it is dropped and not read as zero.
stacked <- do.call(rbind, lapply(STAGES, function(s) {
  keep <- !is.na(units[[s]]) & trimws(as.character(units[[s]])) != ""
  if (!any(keep)) return(NULL)
  data.frame(units[keep, c("session_id", "participant_id", "condition",
                           "unit_index", "scenario_title", "analytical_position")],
             stage = s, attained = as.integer(units[[s]][keep]),
             stringsAsFactors = FALSE)
}))
stacked$stage <- relevel(factor(stacked$stage), ref = STAGE_REF)
stacked$unit_identity <- factor(paste(stacked$scenario_title,
                                      stacked$unit_index, sep = "|"))
# The seven-point ratings live at interaction level; H2a puts one on each row.
for (col in c("post_trust", "post_outcome_confidence")) {
  stacked[[col]] <- scen[[col]][match(stacked$session_id, scen$session_id)]
}

term <- function(model, name, ordinal = FALSE) {
  co <- if (ordinal) summary(model)$coefficients else summary(model)$coefficients$cond
  if (!name %in% rownames(co)) stop("term not found: ", name)
  r <- co[name, ]
  se <- as.numeric(r[2])
  list(estimate = as.numeric(r[1]), se = se, p = as.numeric(r[4]),
       lo = as.numeric(r[1]) - 1.959964 * se,
       hi = as.numeric(r[1]) + 1.959964 * se)
}

record <- function(id, outcome, model_label, t, n, note = "") {
  data.frame(component = id, outcome = outcome, model = model_label,
             ratio = exp(t$estimate), ci_low = exp(t$lo), ci_high = exp(t$hi),
             estimate_log = t$estimate, se = t$se, p = t$p, n_obs = n,
             note = note, stringsAsFactors = FALSE)
}

# ---- repair: Poisson unless the boundary LRT rejects ----
fit_repair <- function(ref) {
  d <- prep(scen, ref)
  f <- repair_post_boundary ~ condition + scenario_title + analytical_position +
    (1 | participant_id)
  pois <- glmmTMB(f, data = d, family = poisson)
  nb <- glmmTMB(f, data = d, family = nbinom2)
  lr <- 2 * (as.numeric(logLik(nb)) - as.numeric(logLik(pois)))
  p_disp <- if (lr > 0) 0.5 * pchisq(lr, 1, lower.tail = FALSE) else 1
  if (p_disp < 0.05) list(m = nb, label = sprintf("NB GLMM (LRT p=%.3g)", p_disp))
  else list(m = pois, label = sprintf("Poisson GLMM (LRT p=%.3g)", p_disp))
}

# ---- uptake, optionally with a rating covariate for H2a ----
fit_uptake <- function(ref, extra = NULL) {
  d <- prep(stacked, ref)
  rhs <- "condition + stage + analytical_position + unit_identity"
  if (!is.null(extra)) rhs <- paste(rhs, "+", extra)
  f <- as.formula(paste("attained ~", rhs,
                        "+ (1 | participant_id) + (1 | session_id)"))
  glmmTMB(f, data = d, family = binomial)
}

rows <- list()

r_nat <- fit_repair(CONDITION_REF)
rows$h1b_repair <- record("h1b_repair", "post-transition repair", r_nat$label,
                          term(r_nat$m, "conditionvc_deactivation"), nrow(scen))
r_conv <- fit_repair("stable_converted")
rows$h1a_repair <- record("h1a_repair", "post-transition repair", r_conv$label,
                          term(r_conv$m, "conditionvc_activation"), nrow(scen))
rows$secondary_stable_repair <- record(
  "secondary_stable_repair", "post-transition repair", r_nat$label,
  term(r_nat$m, "conditionstable_converted"), nrow(scen),
  "secondary; not in the Holm family")

u_nat <- fit_uptake(CONDITION_REF)
rows$h1b_uptake <- record("h1b_uptake", "indicator attainment", "logistic GLMM",
                          term(u_nat, "conditionvc_deactivation"), nrow(stacked))
u_conv <- fit_uptake("stable_converted")
rows$h1a_uptake <- record("h1a_uptake", "indicator attainment", "logistic GLMM",
                          term(u_conv, "conditionvc_activation"), nrow(stacked))
rows$secondary_stable_uptake <- record(
  "secondary_stable_uptake", "indicator attainment", "logistic GLMM",
  term(u_nat, "conditionstable_converted"), nrow(stacked),
  "secondary; not in the Holm family")

for (v in c("post_trust", "post_outcome_confidence")) {
  m <- fit_uptake(CONDITION_REF, extra = v)
  rows[[paste0("h2a_", v)]] <- record(
    paste0("h2a_", sub("^post_", "", v)), "indicator attainment",
    "logistic GLMM", term(m, v), sum(!is.na(stacked[[v]])),
    "odds ratio per scale point")
}

for (v in c("post_effort", "post_frustration")) {
  d <- prep(scen, CONDITION_REF)
  d <- d[!is.na(d[[v]]), ]
  d$y <- factor(d[[v]], ordered = TRUE)
  m <- clmm(y ~ repair_total + condition + scenario_title + analytical_position +
              (1 | participant_id), data = d, Hess = TRUE)
  rows[[paste0("h2b_", v)]] <- record(
    paste0("h2b_", sub("^post_", "", v)), v, "cumulative-link mixed model",
    term(m, "repair_total", ordinal = TRUE), nrow(d),
    "odds ratio per repair move")
}

components <- do.call(rbind, rows)
rownames(components) <- NULL

# Intersection-union: a hypothesis needs both of its components, so its p is
# the larger. Holm adjusts across the four, and the secondaries stay outside.
PAIRS <- list(H1a = c("h1a_repair", "h1a_uptake"),
              H1b = c("h1b_repair", "h1b_uptake"),
              H2a = c("h2a_trust", "h2a_outcome_confidence"),
              H2b = c("h2b_effort", "h2b_frustration"))
raw <- vapply(PAIRS, function(p) max(components$p[match(p, components$component)]),
              numeric(1))
hyp <- data.frame(hypothesis = names(PAIRS), p = raw,
                  p_holm = p.adjust(raw, method = "holm"),
                  components = vapply(PAIRS, paste, character(1), collapse = " + "),
                  stringsAsFactors = FALSE)
rownames(hyp) <- NULL

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(components, file.path(out_dir, "confirmatory_r_components.csv"),
          row.names = FALSE)
write.csv(hyp, file.path(out_dir, "confirmatory_r_hypotheses.csv"),
          row.names = FALSE)

cat("=== components (ratios with 95% Wald intervals)\n")
print(components[, c("component", "model", "ratio", "ci_low", "ci_high",
                     "p", "n_obs")], row.names = FALSE, digits = 3)
cat("\n=== hypotheses (intersection-union, Holm over the four)\n")
print(hyp, row.names = FALSE, digits = 3)
cat("\nwritten:", file.path(out_dir, "confirmatory_r_components.csv"), "\n")
