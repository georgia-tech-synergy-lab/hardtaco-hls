PYTHON  := /usr/local/bin/python3

MKDIR :=  mkdir -p
CP := cp -r
RM := rm -rf

DESIGNS_PATH=./designs
SIM_RUN_PATH=./sim


sim: clean
	$(MKDIR) $(SIM_RUN_PATH)
	$(CP) $(DESIGNS_PATH)/* $(SIM_RUN_PATH)
	$(PYTHON) sim_init.py -p sim_param.cfg


clean:
	find . -name '*.pyc' -exec rm --force {} +
	$(RM) $(SIM_RUN_PATH) *csv
