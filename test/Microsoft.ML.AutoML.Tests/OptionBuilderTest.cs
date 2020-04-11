﻿using System;
using System.Collections.Generic;
using System.Text;
using FluentAssertions;
using Microsoft.ML.AutoML.AutoPipeline.Sweeper;
using Xunit;

namespace Microsoft.ML.AutoML.Tests
{
    public class OptionBuilderTest
    {
        [Fact]
        public void OptionBuilder_should_create_default_option()
        {
            var builder = new TestOptionBuilder();
            var option = builder.CreateDefaultOption();
            option.IntOption.Should().Equals(1);
            option.FloatOption.Should().Equals(1f);
            option.StringOption.Should().Equals(string.Empty);
        }

        [Fact]
        public void OptionBuilder_should_build_optoin()
        {
            var builder = new TestOptionBuilder();
            var input = new SweeperOutput()
            {
                {"IntOption", 2 },
                {"FloatOption", 2f },
                {"StringOption", "2" },
            };

            var option = builder.BuildOption(input);
            option.IntOption.Should().Equals(2);
            option.FloatOption.Should().Equals(2f);
            option.StringOption.Should().Equals("2");
        }

        [Fact]
        public void OptionBuilder_should_work_with_random_sweeper()
        {
            var builder = new TestOptionBuilder();
            var randomSweeper = builder.BuildRandomSweeper(10);

            foreach ( var sweeperOutput in randomSweeper)
            {
                var option = builder.BuildOption(sweeperOutput);
                option.IntOption
                      .Should()
                      .BeLessOrEqualTo(100)
                      .And
                      .BeGreaterOrEqualTo(0);

                option.FloatOption
                      .Should()
                      .BeLessOrEqualTo(100f)
                      .And
                      .BeGreaterOrEqualTo(0f);

                option.StringOption
                      .Should()
                      .BeOneOf(new string[] { "str1", "str2", "str3", "str4" });
            }
        }


        private class TestOption
        {
            public int IntOption { get; set; } = 1;

            public float FloatOption { get; set; } = 1f;

            public string StringOption { get; set; } = string.Empty;
        }

        private class TestOptionBuilder : OptionBuilder<TestOption>
        {
            [Parameter(0, 100)]
            public int IntOption { get; set; }

            [Parameter(0f, 100f)]
            public float FloatOption { get; set; }

            [Parameter(new string[] { "str1", "str2", "str3", "str4" })]
            public string StringOption { get; set; }
        }
    }


}
