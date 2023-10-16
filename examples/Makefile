FORCE_CPU?=False
run_anomaly_detection: data/art_daily_small_noise.csv data/art_daily_jumpsup.csv
	python3 anomaly_detection.py --force_cpu=$(FORCE_CPU)

data/art_daily_small_noise.csv:
	@mkdir -p data
	wget -O $@ https://raw.githubusercontent.com/numenta/NAB/master/data/artificialNoAnomaly/$(notdir $@)

data/art_daily_jumpsup.csv:
	@mkdir -p data
	wget -O $@ https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/$(notdir $@)
