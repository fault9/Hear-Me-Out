# Condition contrasts over the scripted soundboard audit, in two parts.
#
#   Rscript audit_models.R                      both complete script-mode runs
#   Rscript audit_models.R a/audit_turns.csv …  named run tables instead
#
# response_latency_ms is undefined exactly when the assistant began before the
# clip ended, so the missingness is the premature-onset outcome. Modelling gap
# duration alone would condition on it, and the turns it drops are the fast
# ones, so the effect is split: whether the assistant waited, then how long it
# waited given that it did. The aborted run and the 9-arm probe are excluded.

suppressMessages({
  library(lme4)
})

DATA_ROOT <- Sys.getenv("STUDY_DATA_ROOT", "/workspace/data")
COMPLETE_RUNS <- c("20260807T234553Z_ae22f681", "20260808T171452Z_ad8ddce2")

args <- commandArgs(trailingOnly = TRUE)
paths <- if (length(args)) args else
  file.path(DATA_ROOT, "audit", COMPLETE_RUNS, "audit_turns.csv")
missing <- paths[!file.exists(paths)]
if (length(missing)) {
  stop("run audit_postprocess first; no table at: ", paste(missing, collapse = ", "))
}

# Pooled here so the run label always matches the directory rows came from.
d <- do.call(rbind, lapply(paths, function(p) {
  rows <- read.csv(p, stringsAsFactors = FALSE)
  rows$audit_run <- basename(dirname(p))
  rows
}))

# The pipeline writes an empty list, not an empty string, for a clean turn.
invalid <- trimws(d$technical_invalid_reasons)
d <- d[invalid == "" | invalid == "[]", ]

as_bool <- function(x) tolower(trimws(as.character(x))) %in% c("true", "1")
d$premature      <- as_bool(d$premature_assistant_onset)
d$manipulation   <- factor(d$manipulation)
d$source_speaker <- factor(d$source_speaker)
d$audit_run      <- factor(d$audit_run)
# Replay and script position carry random intercepts; run stays fixed, since
# two levels cannot support a variance estimate.
d$replay   <- factor(paste(d$audit_run, d$rep, sep = ":"))
d$position <- factor(d$turn)

cat("turns:", nrow(d), " replays:", nlevels(d$replay),
    " runs:", nlevels(d$audit_run), "\n\n")
cat("premature-onset rate by cell\n")
print(round(with(d, tapply(premature, list(source_speaker, manipulation), mean)), 4))

wald <- function(model) {
  round(exp(cbind(estimate = fixef(model),
                  confint(model, method = "Wald", parm = "beta_"))), 3)
}

# ---- part 1: did the assistant wait at all ----
m1 <- glmer(premature ~ manipulation * source_speaker + audit_run +
              (1 | replay) + (1 | position),
            data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
cat("\n== part 1: premature onset (logistic, every turn)\n")
print(round(summary(m1)$coefficients, 4))
cat("\nodds ratios, 95% Wald CI\n")
print(wald(m1))

# ---- part 2: how long it waited, given it waited ----
# Logged, so condition terms read as multiplicative.
d$gap <- suppressWarnings(as.numeric(d$positive_response_gap_ms))
w <- d[!d$premature & !is.na(d$gap) & d$gap > 0, ]
cat("\n== part 2: response gap given the assistant waited (n =", nrow(w), ")\n")
m2 <- lmer(log(gap) ~ manipulation * source_speaker + audit_run +
             (1 | replay) + (1 | position), data = w)
print(round(summary(m2)$coefficients, 4))
cat("\nmultiplicative effect on gap duration, 95% Wald CI\n")
print(wald(m2))

cat("\nPart 2 is conditional on a positive gap and its conditioning set differs",
    "\nby condition, per part 1. Report the parts together, never part 2 alone.\n")
