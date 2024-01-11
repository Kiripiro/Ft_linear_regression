ARGS =

all: training prediction

training:
	@python3 training.py $(ARGS)

prediction:
	@python3 prediction.py

clean:
	@rm -rf ./images/*.png
	@rm -rf ./images/*.gif

fclean: clean
		@rm -rf ./images
		@rm -rf __pycache__
		@rm raw_thetas.csv

re: fclean all

.PHONY: all training prediction clean fclean re