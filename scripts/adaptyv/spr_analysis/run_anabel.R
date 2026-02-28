# Run anabel MCK analysis on SPR data with reference subtraction
# Automatically detects all sample IDs, replicates, and concentrations
#
# Protocol: 60s baseline -> 300s association -> 600s dissociation
# Data is baseline-subtracted; association starts at t=0, dissociation at t=300
#
# Reference correction: blank sensorgrams (from reference channel) are subtracted
# from sample sensorgrams to remove bulk refractive index effects.
#
# Fit quality: chi2 (mean squared residual) and R2 computed from fit_data

.libPaths(c("~/R/libs", .libPaths()))
library(anabel)

script_dir <- getwd()  # run from the spr_2383 directory
raw_dir <- file.path(script_dir, "raw_data")
blank_dir <- file.path(script_dir, "blanks", "raw_data")
out_dir <- file.path(script_dir, "anabel_output")
dir.create(out_dir, showWarnings = FALSE)

# --- Load run mapping and blank mapping ---
run_map <- read.csv(file.path(script_dir, "blanks", "run_mapping.csv"),
                    stringsAsFactors = FALSE)

# Map from run number to blank file
# run_mapping "runs" column corresponds to "read" in read_data.csv
blank_map <- list(
  "1" = "blank_1_1.csv",
  "2" = "blank_2_2.csv",
  "3" = "blank_3_3.csv",
  "4" = "blank_5_4.csv"
)

# Pre-load blank data
blanks <- list()
for (run_id in names(blank_map)) {
  bf <- file.path(blank_dir, blank_map[[run_id]])
  if (file.exists(bf)) {
    blanks[[run_id]] <- read.csv(bf)
    cat(sprintf("Loaded blank for run %s: %s (mean=%.3f RU)\n",
                run_id, blank_map[[run_id]], mean(blanks[[run_id]]$y)))
  }
}

# --- Auto-detect all sample IDs from raw_data ---
all_raw_files <- list.files(raw_dir, pattern = "^[0-9]+_\\d+_.+\\.csv$")
sample_ids <- sort(unique(sub("_\\d+_.*", "", all_raw_files)))
cat(sprintf("\nFound %d sample IDs: %s\n", length(sample_ids), paste(sample_ids, collapse = ", ")))

# --- Collect all results ---
all_results <- list()

for (sid in sample_ids) {
  cat(sprintf("\n%s\n  Sample %s\n%s\n",
              paste(rep("=", 60), collapse = ""), sid,
              paste(rep("=", 60), collapse = "")))

  # Detect available files for this sample
  sample_files <- list.files(raw_dir, pattern = sprintf("^%s_", sid))
  if (length(sample_files) == 0) {
    cat("  No data files found, skipping.\n")
    next
  }

  # Parse replicate and concentration from filenames
  parsed <- regmatches(sample_files,
                       regexec(sprintf("^%s_(\\d+)_(.+)\\.csv$", sid), sample_files))
  rep_conc <- data.frame(
    rep = as.integer(sapply(parsed, `[`, 2)),
    conc = as.numeric(sapply(parsed, `[`, 3)),
    file = sample_files,
    stringsAsFactors = FALSE
  )
  available_reps <- sort(unique(rep_conc$rep))

  sample_outdir <- file.path(out_dir, sid)
  dir.create(sample_outdir, showWarnings = FALSE)

  for (rep in available_reps) {
    rep_files <- rep_conc[rep_conc$rep == rep, ]
    concs <- sort(rep_files$conc)

    cat(sprintf("\n--- Replicate %d (%d conc: %s nM) ---\n",
                rep, length(concs), paste(concs, collapse = ", ")))

    if (length(concs) < 2) {
      cat("  Too few concentrations for MCK, skipping.\n")
      next
    }

    # Look up the run number for this sample/replicate
    run_row <- run_map[run_map$name == sid & run_map$replicate == rep, ]
    run_id <- NULL
    blank_data <- NULL
    if (nrow(run_row) > 0) {
      run_id <- as.character(run_row$runs[1])
      if (run_id %in% names(blanks)) {
        blank_data <- blanks[[run_id]]
        cat(sprintf("  Using blank for run %s (%s)\n", run_id, blank_map[[run_id]]))
      } else {
        cat(sprintf("  No blank available for run %s\n", run_id))
      }
    } else {
      cat("  No run mapping found, skipping reference correction\n")
    }

    # Read all concentration files
    dfs <- list()
    for (conc in concs) {
      conc_str <- ifelse(conc == floor(conc), sprintf("%.1f", conc), as.character(conc))
      f <- file.path(raw_dir, sprintf("%s_%d_%s.csv", sid, rep, conc_str))
      d <- read.csv(f)
      dfs[[as.character(conc)]] <- d
    }

    # Create common time grid
    t_min <- max(sapply(dfs, function(x) min(x$t)))
    t_max <- min(sapply(dfs, function(x) max(x$t)))
    if (!is.null(blank_data)) {
      t_max <- min(t_max, max(blank_data$t))
    }
    common_t <- seq(ceiling(t_min), floor(t_max), by = 1)

    cat(sprintf("  Time range: %d - %d s, %d points\n",
                min(common_t), max(common_t), length(common_t)))

    # Interpolate blank to common time grid
    blank_interp <- NULL
    if (!is.null(blank_data)) {
      blank_interp <- approx(blank_data$t, blank_data$y, xout = common_t)$y
    }

    # Build wide data frame with reference subtraction
    wide <- data.frame(Time = common_t)
    for (conc in concs) {
      d <- dfs[[as.character(conc)]]
      interp <- approx(d$t, d$y, xout = common_t)$y
      if (!is.null(blank_interp)) {
        interp <- interp - blank_interp
      }
      col_name <- paste0("Conc_", conc, "_nM")
      wide[[col_name]] <- interp
    }

    conc_M <- concs * 1e-9

    # Run anabel MCK fitting with debug_mode to get fit values
    rep_outdir <- file.path(sample_outdir, sprintf("rep%d", rep))
    dir.create(rep_outdir, showWarnings = FALSE)

    result <- tryCatch({
      run_anabel(
        input = wide,
        tstart = 0,
        tass = 0,
        tdiss = 300,
        conc = conc_M,
        method = "MCK",
        quiet = TRUE,
        outdir = rep_outdir,
        generate_output = "all",
        debug_mode = TRUE
      )
    }, error = function(e) {
      cat("  Error:", conditionMessage(e), "\n")
      NULL
    })

    key <- sprintf("%s_rep%d", sid, rep)
    if (!is.null(result) && nrow(result$kinetics) > 0) {
      k <- result$kinetics

      # Compute chi2 and R2 from fit_data
      chi2 <- NA
      r2 <- NA
      if (!is.null(result$fit_data) && nrow(result$fit_data) > 0) {
        fd <- result$fit_data
        obs <- fd$Response
        pred <- fd$fit
        residuals <- obs - pred
        n_params <- 4  # ka, kd, Rmax, y_offset (global MCK params)
        n_points <- length(residuals)
        chi2 <- sum(residuals^2) / (n_points - n_params)
        ss_res <- sum(residuals^2)
        ss_tot <- sum((obs - mean(obs))^2)
        r2 <- 1 - ss_res / ss_tot
      }

      cat(sprintf("  KD    = %.2e M (%.1f nM)\n", k$KD, k$KD * 1e9))
      cat(sprintf("  kon   = %.2e M-1 s-1\n", k$kass))
      cat(sprintf("  koff  = %.4f s-1\n", k$kdiss))
      cat(sprintf("  Rmax  = %.2f RU\n", k$Rmax))
      cat(sprintf("  chi2  = %.4f RU^2\n", chi2))
      cat(sprintf("  R2    = %.4f\n", r2))
      cat(sprintf("  Quality: %s\n", k$FittingQ))

      all_results[[key]] <- data.frame(
        Sample = sid, Replicate = rep,
        KD_nM = k$KD * 1e9, kon = k$kass, koff = k$kdiss,
        Rmax = k$Rmax, chi2 = chi2, R2 = r2,
        Quality = as.character(k$FittingQ),
        n_conc = length(concs), run = ifelse(is.null(run_id), NA, run_id),
        ref_corrected = !is.null(blank_data),
        stringsAsFactors = FALSE
      )
    } else {
      cat("  Fitting failed.\n")
      all_results[[key]] <- data.frame(
        Sample = sid, Replicate = rep,
        KD_nM = NA, kon = NA, koff = NA,
        Rmax = NA, chi2 = NA, R2 = NA,
        Quality = "Failed",
        n_conc = length(concs), run = ifelse(is.null(run_id), NA, run_id),
        ref_corrected = !is.null(blank_data),
        stringsAsFactors = FALSE
      )
    }
  }
}

# --- Summary ---
cat(sprintf("\n\n%s\n  SUMMARY\n%s\n",
            paste(rep("=", 60), collapse = ""),
            paste(rep("=", 60), collapse = "")))
summary_df <- do.call(rbind, all_results)
rownames(summary_df) <- NULL
print(summary_df)

# Save summary CSV
summary_file <- file.path(out_dir, "anabel_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to: %s\n", summary_file))

# --- Pivot table with quality metrics ---
library(tidyr)
library(dplyr, warn.conflicts = FALSE)

pivot <- summary_df %>%
  mutate(
    fit_flag = case_when(
      Quality == "Failed" ~ "FAILED",
      is.na(KD_nM) ~ "FAILED",
      Rmax < 0 ~ "bad",
      Rmax > 100 ~ "bad",
      TRUE ~ "ok"
    ),
    cell = case_when(
      fit_flag == "FAILED" ~ "FAILED",
      fit_flag == "bad" ~ sprintf("%.1f *", KD_nM),
      TRUE ~ sprintf("%.1f", KD_nM)
    ),
    chi2_fmt = ifelse(is.na(chi2), "NA", sprintf("%.2f", chi2)),
    r2_fmt = ifelse(is.na(R2), "NA", sprintf("%.3f", R2)),
    Replicate = paste0("rep", Replicate)
  )

kd_pivot <- pivot %>%
  select(Sample, Replicate, cell) %>%
  pivot_wider(names_from = Replicate, values_from = cell)

chi2_pivot <- pivot %>%
  select(Sample, Replicate, chi2_fmt) %>%
  pivot_wider(names_from = Replicate, values_from = chi2_fmt, names_prefix = "chi2_")

r2_pivot <- pivot %>%
  select(Sample, Replicate, r2_fmt) %>%
  pivot_wider(names_from = Replicate, values_from = r2_fmt, names_prefix = "R2_")

rmax_pivot <- pivot %>%
  select(Sample, Replicate, Rmax) %>%
  mutate(Rmax = ifelse(is.na(Rmax), "NA", sprintf("%.1f", Rmax))) %>%
  pivot_wider(names_from = Replicate, values_from = Rmax, names_prefix = "Rmax_")

full_pivot <- kd_pivot %>%
  left_join(rmax_pivot, by = "Sample") %>%
  left_join(chi2_pivot, by = "Sample") %>%
  left_join(r2_pivot, by = "Sample") %>%
  mutate(Sample_int = as.integer(Sample)) %>%
  arrange(Sample_int) %>%
  select(-Sample_int)

pivot_file <- file.path(out_dir, "anabel_kd_pivot.csv")
write.csv(full_pivot, pivot_file, row.names = FALSE)
cat(sprintf("\nPivot table saved to: %s\n", pivot_file))
