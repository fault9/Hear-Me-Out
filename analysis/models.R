# Confirmatory models per the method chapter's Analysis section.
#
#   Rscript models.R --permute        pipeline smoke test (labels shuffled
#                                     within participant; results meaningless)
#   Rscript models.R --exploratory    real labels on incomplete data: is the
#                                     model estimable at all? Preliminary,
#                                     repeatable, and not the confirmatory run
#   Rscript models.R --confirmatory   the once-only real run
#
# Two primary contrasts, each holding the post-transition voice state constant:
#   A: vc_activation   vs stable_converted
#   B: vc_deactivation vs stable_natural
# Co-primary outcomes: grounding (logistic GLMM) and repair_post_boundary
# (Poisson GLMM, negative-binomial if overdispersed). The four tests form one
# Holm-adjusted family.
#
# GROUNDING_LEVEL selects the grounding outcome. The interaction-level
# demonstrated_grounding proved unidentifiable on collected data - 3 successes
# across 45 interactions, complete separation, intervals thousands of log-odds
# wide - so the prespecified fallback moves it to the critical unit.
#
#   stacked  (default) one model over all four uptake stages, each unit
#            contributing a row per observable stage, with stage as a fixed
#            factor and random intercepts for participant and unit. The stages
#            are complementary rather than nested - operational use occurs
#            without acknowledgement, and exceeds claimed record update - so
#            they are not a cascade to be modelled sequentially. Pooling them
#            estimates one condition effect on uptake instead of splitting the
#            same observations across four underpowered tests, and the
#            condition-by-stage term reports whether that effect is uniform.
#   unit     a single stage, named by GROUNDING_OUTCOME
#   scenario the original interaction-level demonstrated_grounding

suppressMessages({
  library(lme4)
  library(glmmTMB)
})

args <- commandArgs(trailingOnly = TRUE)
permute <- "--permute" %in% args
exploratory <- "--exploratory" %in% args
confirmatory <- "--confirmatory" %in% args
sensitivity <- "--complete-technical-sensitivity" %in% args
if (sum(permute, exploratory, confirmatory) != 1) {
  stop("pass exactly one of --permute, --exploratory or --confirmatory")
}

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
frames <- file.path(here, "output", "frames")
frame_file <- if (sensitivity) {
  "scenario_level_sensitivity_complete_technical.csv"
} else {
  "scenario_level.csv"
}
dat <- read.csv(file.path(frames, frame_file), stringsAsFactors = FALSE)
if (sensitivity) cat("(sensitivity frame: excludes participants with any technically invalid analytical attempt)\n")

grounding_level <- Sys.getenv("GROUNDING_LEVEL", "stacked")
grounding_outcome <- Sys.getenv("GROUNDING_OUTCOME", "incorporation")
unit_file <- if (sensitivity) {
  "unit_level_sensitivity_complete_technical.csv"
} else {
  "unit_level.csv"
}
units <- read.csv(file.path(frames, unit_file), stringsAsFactors = FALSE)

read_optional <- function(name) {
  path <- file.path(frames, name)
  if (file.exists(path)) read.csv(path, stringsAsFactors = FALSE) else NULL
}
events <- read_optional("turn_events_certified.csv")
gaps <- read_optional("turn_gaps_certified.csv")

UPTAKE_STAGES <- c("acknowledgement", "update_claim", "incorporation",
                   "retention")

stack_uptake <- function(u) {
  rows <- lapply(UPTAKE_STAGES, function(stage) {
    data.frame(participant_id = u$participant_id, session_id = u$session_id,
               condition = u$condition, scenario_title = u$scenario_title,
               analytical_position = u$analytical_position,
               unit_key = paste(u$session_id, u$unit_index, sep = ":"),
               stage = stage, grounding = parse_binary(u[[stage]], stage),
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  # A stage the coder could not judge is missing, not a failure.
  out[!is.na(out$grounding), ]
}

parse_binary <- function(values, field) {
  text <- tolower(trimws(as.character(values)))
  result <- rep(NA_integer_, length(text))
  result[text %in% c("1", "true", "yes")] <- 1L
  result[text %in% c("0", "false", "no")] <- 0L
  invalid <- !is.na(text) & nzchar(text) & is.na(result)
  if (any(invalid)) {
    stop(sprintf("%s contains invalid non-missing value(s): %s", field,
                 paste(unique(text[invalid]), collapse = ", ")))
  }
  result
}

if (grounding_level == "stacked") {
  units <- stack_uptake(units)
  units$participant_id <- factor(units$participant_id)
  units$unit_key <- factor(units$unit_key)
  units$scenario_title <- factor(units$scenario_title)
  units$stage <- factor(units$stage, levels = UPTAKE_STAGES)
  units$position <- suppressWarnings(as.integer(units$analytical_position))
  cat(sprintf("grounding: stacked over %s (%d observations, %d units)\n",
              paste(UPTAKE_STAGES, collapse = "/"), nrow(units),
              nlevels(units$unit_key)))
} else if (grounding_level == "unit") {
  units$grounding <- parse_binary(units[[grounding_outcome]], grounding_outcome)
  units$participant_id <- factor(units$participant_id)
  units$scenario_title <- factor(units$scenario_title)
  units$position <- suppressWarnings(as.integer(units$analytical_position))
  cat(sprintf("grounding: unit-level %s (%d units)\n",
              grounding_outcome, sum(!is.na(units$grounding))))
} else {
  cat("grounding: interaction-level demonstrated_grounding\n")
}
dat$grounding <- parse_binary(dat$demonstrated_grounding,
                              "demonstrated_grounding")
dat$repairs_post <- suppressWarnings(as.integer(dat$repair_post_boundary))
dat$participant_id <- factor(dat$participant_id)
dat$scenario_title <- factor(dat$scenario_title)
dat$position <- suppressWarnings(as.integer(dat$analytical_position))

if (permute) {
  cat("=== PERMUTED SMOKE RUN — condition labels shuffled; results are meaningless ===\n")
  set.seed(20260805)
  dat$condition <- ave(dat$condition, dat$participant_id,
                       FUN = function(x) sample(x))
  units$condition <- ave(units$condition, units$participant_id,
                         FUN = function(x) sample(x))
} else if (exploratory) {
  cat("=== EXPLORATORY RUN — real labels, incomplete data ===\n")
  cat("Estimates are preliminary and will move. Read the diagnostics: they\n")
  cat("say whether these models can be fitted, which is decidable now. The\n")
  cat("confirmatory run remains unspent.\n")
} else {
  cat("=== CONFIRMATORY RUN — this is the once-only real analysis ===\n")
}

# The two primary contrasts hold the post-transition voice constant. The
# third asks whether a converted voice matters at all, which is the baseline
# the other two are read against; it is secondary and stays outside the
# prespecified Holm family.
contrasts <- list(
  A = c(treated = "vc_activation", reference = "stable_converted"),
  B = c(treated = "vc_deactivation", reference = "stable_natural")
)
secondary_contrasts <- list(
  C = c(treated = "stable_converted", reference = "stable_natural")
)

# An episode in a certified session is not a verified episode. Counting only
# what a listener confirmed keeps the automatic nominations - 45% of which did
# not survive verification - out of the models.
turn_counts <- function(events, sessions) {
  if (is.null(events) || !nrow(events)) return(NULL)
  judged <- events[nzchar(as.character(events$verifier_initials)), ]
  if (!nrow(judged)) return(NULL)
  judged$ov <- parse_binary(judged$verified_overlap, "verified_overlap")
  judged$pre <- parse_binary(judged$verified_assistant_premature_onset,
                             "verified_assistant_premature_onset")
  judged$bi <- parse_binary(judged$verified_participant_barge_in,
                            "verified_participant_barge_in")
  per <- do.call(rbind, lapply(split(judged, judged$session_id), function(g)
    data.frame(session_id = g$session_id[1],
               verified_overlaps = sum(g$ov, na.rm = TRUE),
               verified_premature = sum(g$pre, na.rm = TRUE),
               verified_barge_ins = sum(g$bi, na.rm = TRUE),
               stringsAsFactors = FALSE)))
  # A certified session with nothing confirmed contributes zeros, not a gap in
  # the frame; dropping it would condition the count on having any event.
  base <- data.frame(session_id = unique(events$session_id),
                     stringsAsFactors = FALSE)
  out <- merge(base, per, by = "session_id", all.x = TRUE)
  for (field in c("verified_overlaps", "verified_premature",
                  "verified_barge_ins")) {
    out[[field]][is.na(out[[field]])] <- 0L
  }
  merge(sessions[, c("session_id", "participant_id", "condition",
                     "scenario_title", "position")], out, by = "session_id")
}

turns <- turn_counts(events, dat)

fit_one <- function(sub, outcome, count_field = "repairs_post") {
  sub$cond <- factor(sub$condition,
                     levels = c(sub$reference[1], sub$treated[1]))
  if (outcome == "grounding") {
    stacked <- "stage" %in% names(sub)
    form <- if (stacked) {
      grounding ~ cond + stage + scenario_title + position +
        (1 | participant_id) + (1 | unit_key)
    } else {
      grounding ~ cond + scenario_title + position + (1 | participant_id)
    }
    m <- glmer(form, data = sub, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
    co <- summary(m)$coefficients
    row <- co[grep("^cond", rownames(co)), , drop = FALSE]
    return(list(estimate = row[1, 1], se = row[1, 2], p = row[1, 4],
                model = "logistic GLMM", n = nrow(sub),
                outcome_rate = mean(sub$grounding),
                outcome_events = sum(sub$grounding),
                re_sd = sqrt(as.numeric(VarCorr(m)$participant_id)),
                stages = if (stacked) nlevels(droplevels(sub$stage)) else 1L,
                singular = lme4::isSingular(m),
                dispersion = NA_real_,
                note = paste(m@optinfo$conv$lme4$messages, collapse = "; ")))
  }
  # Counts: Poisson first; refit negative-binomial when overdispersed
  # (Pearson chi-square / df > 1.5).
  form <- as.formula(paste(count_field,
                           "~ cond + scenario_title + position + (1 | participant_id)"))
  m <- glmmTMB(form, data = sub, family = poisson)
  pearson <- sum(residuals(m, type = "pearson")^2)
  ratio <- pearson / df.residual(m)
  label <- "Poisson GLMM"
  if (is.finite(ratio) && ratio > 1.5) {
    m <- update(m, family = nbinom2)
    label <- sprintf("NB GLMM (dispersion %.2f)", ratio)
  }
  co <- summary(m)$coefficients$cond
  row <- co[grep("^cond", rownames(co)), , drop = FALSE]
  re <- tryCatch(sqrt(as.numeric(VarCorr(m)$cond$participant_id)),
                 error = function(e) NA_real_)
  list(estimate = row[1, 1], se = row[1, 2], p = row[1, 4],
       model = label, n = nrow(sub),
       outcome_rate = mean(sub[[count_field]]),
       outcome_events = sum(sub[[count_field]]), stages = 1L,
       re_sd = re, singular = isTRUE(re < 1e-4), dispersion = ratio,
       note = if (isTRUE(m$sdr$pdHess)) "" else "Hessian not positive definite")
}

all_contrasts <- c(contrasts, secondary_contrasts)
specs <- list()
for (name in names(contrasts)) {
  for (outcome in c("grounding", "repairs_post")) {
    specs[[length(specs) + 1]] <- list(name = name, outcome = outcome,
                                       family = "primary")
  }
}
for (name in names(secondary_contrasts)) {
  for (outcome in c("grounding", "repairs_post")) {
    specs[[length(specs) + 1]] <- list(name = name, outcome = outcome,
                                       family = "secondary")
  }
}
if (!is.null(turns)) {
  for (name in names(all_contrasts)) {
    for (outcome in c("verified_premature", "verified_overlaps")) {
      specs[[length(specs) + 1]] <- list(name = name, outcome = outcome,
                                         family = "secondary")
    }
  }
}

results <- data.frame()
for (spec in specs) {
  pair <- all_contrasts[[spec$name]]
  outcome <- spec$outcome
  counted <- outcome != "grounding"
  frame <- if (outcome == "grounding" && grounding_level != "scenario") {
    units
  } else if (startsWith(outcome, "verified_")) {
    turns
  } else {
    dat
  }
  sub <- frame[frame$condition %in% pair, ]
  sub$treated <- pair[["treated"]]
  sub$reference <- pair[["reference"]]
  sub <- sub[!is.na(sub$position), ]
  keep <- if (counted) !is.na(sub[[outcome]]) else !is.na(sub$grounding)
  fit <- tryCatch(fit_one(sub[keep, ], if (counted) "count" else "grounding",
                          count_field = outcome),
                  error = function(e) list(estimate = NA, se = NA, p = NA,
                                           model = paste("FAILED:", conditionMessage(e)),
                                           n = sum(keep), outcome_rate = NA,
                                           outcome_events = NA, re_sd = NA,
                                           stages = NA,
                                           singular = NA, dispersion = NA,
                                           note = conditionMessage(e)))
  results <- rbind(results, data.frame(
    family = spec$family,
    contrast = sprintf("%s: %s vs %s", spec$name, pair[["treated"]],
                       pair[["reference"]]),
    outcome = if (outcome != "grounding") {
      outcome
    } else if (grounding_level == "stacked") {
      "grounding/stacked"
    } else if (grounding_level == "unit") {
      paste0("grounding/", grounding_outcome)
    } else {
      outcome
    },
    model = fit$model, estimate = fit$estimate,
    se = fit$se, p = fit$p, n_obs = fit$n,
    ci_low = fit$estimate - 1.96 * fit$se,
    ci_high = fit$estimate + 1.96 * fit$se,
    outcome_rate = fit$outcome_rate, outcome_events = fit$outcome_events,
    stages = fit$stages,
    re_sd = fit$re_sd, singular = fit$singular, dispersion = fit$dispersion,
    note = fit$note,
    n_participants = length(unique(sub$participant_id[keep])),
    permuted = permute))
}

# The prespecified family is the two primary contrasts crossed with the two
# co-primary outcomes. The secondary contrast and the verified turn-taking
# counts are reported unadjusted and labelled, rather than being folded into a
# family that was frozen before they existed.
results$p_holm <- NA_real_
primary <- results$family == "primary"
results$p_holm[primary] <- p.adjust(results$p[primary], method = "holm")
stem <- if (permute) "smoke_results" else if (exploratory) {
  "exploratory_results"
} else {
  "confirmatory_results"
}
if (sensitivity) stem <- paste0(stem, "_sensitivity_complete_technical")
out <- file.path(here, "output", paste0(stem, ".csv"))
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
write.csv(results, out, row.names = FALSE)
show <- c("contrast", "outcome", "model", "estimate", "ci_low", "ci_high",
          "p", "p_holm", "n_obs")
cat("\n=== primary family (Holm-adjusted) ===\n")
print(results[primary, show])
if (any(!primary)) {
  cat("\n=== secondary, unadjusted ===\n")
  print(results[!primary, setdiff(show, "p_holm")])
}

# Whether the design can carry these models is answerable on partial data;
# whether the effect is real is not. A singular random effect, a near-constant
# outcome or a 20-wide confidence interval means the model needs changing, and
# that change has to be made before the confirmatory run rather than after it.
cat("\n=== model diagnostics ===\n")
print(results[, c("family", "contrast", "outcome", "n_obs", "n_participants",
                  "stages", "outcome_events", "outcome_rate", "re_sd",
                  "singular", "dispersion", "note")])
cat("\nRead as: singular TRUE means the participant random effect collapsed;\n")
cat("outcome_rate near 0 or 1 means the logistic model has little to fit;\n")
cat("dispersion over 1.5 is why the repair model switches to negative binomial;\n")
cat("a ci_low/ci_high spanning several log-odds means the data cannot yet\n")
cat("distinguish anything, which is a power statement and not a null result.\n")
# Detector precision is a prerequisite, not an aside: if the automatic gap
# detector is less accurate under one voice than another, every automatic
# turn-taking count carries the manipulation inside it. The sample was
# stratified on condition so this could be checked rather than assumed.
if (!is.null(gaps)) {
  judged <- gaps[nzchar(as.character(gaps$verifier_initials)), ]
  if (nrow(judged)) {
    judged$real <- parse_binary(judged$verified_positive_gap,
                                "verified_positive_gap")
    cat("\n=== response-gap detector precision (verified sample) ===\n")
    per <- do.call(rbind, lapply(split(judged, judged$condition), function(g) {
      n <- sum(!is.na(g$real))
      k <- sum(g$real, na.rm = TRUE)
      # Wilson interval: at these counts a normal approximation would run
      # outside [0, 1].
      z <- 1.96
      phat <- if (n) k / n else NA_real_
      centre <- (k + z^2 / 2) / (n + z^2)
      halfwidth <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) /
        (1 + z^2 / n)
      data.frame(condition = g$condition[1], judged = n, real = k,
                 precision = round(phat, 3),
                 ci_low = round(max(0, centre - halfwidth), 3),
                 ci_high = round(min(1, centre + halfwidth), 3))
    }))
    print(per, row.names = FALSE)
    if (nrow(per) > 1 && min(per$judged) >= 5) {
      test <- tryCatch(
        summary(glmer(real ~ condition + (1 | participant_id), data = judged,
                      family = binomial,
                      control = glmerControl(optimizer = "bobyqa"))),
        error = function(e) NULL)
      if (!is.null(test)) {
        cat("\nprecision by condition (logistic, participant intercept):\n")
        print(round(test$coefficients, 4))
      }
    }
    cat("\nPrecision far below 1 means the automatic gap count is not a\n")
    cat("measurement; differing precision across conditions means it is not\n")
    cat("even a comparable one. Report verified gaps weighted by stratum.\n")
  }
}

cat("\nwritten:", out, "\n")
