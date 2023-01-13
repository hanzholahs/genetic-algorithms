def write_log_file(simlog_path, simlog_lines, **kwargs):
    with open(simlog_path, "a") as log:
        for i, line in enumerate(simlog_lines):
            log.write(line)
            log.write("\n")
            if i == 0:
                for key, value in kwargs.items():
                    log.write(f"\t{key} = {value}")
                    log.write("\n")
                log.write("\n")
        log.write("\n\n")