        parameter: PropertyParameter.Config = QuaternionParameter.Config()
        parameter = RangeParameter.Config(
            BoundaryNormalizer.Config(
                lower=[self.low],
                upper=[self.high],
            ),
            threshold=0.05,
        )

        parameter: PropertyParameter.Config = EuclideanParameter.Config()
        parameter: PropertyParameter.Config = FlipParameter.Config()
        parameter: PropertyParameter.Config = BinaryParameter.Config()
        parameter: PropertyParameter.Config = EuclideanParameter.Config()
