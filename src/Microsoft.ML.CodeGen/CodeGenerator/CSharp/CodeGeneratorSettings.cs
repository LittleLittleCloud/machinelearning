﻿using Microsoft.ML.AutoML;

namespace Microsoft.ML.CodeGen.CSharp
{
    public class CodeGeneratorSettings
    {
        public string LabelName { get; set; }

        public string ModelPath { get; set; }

        public string OutputName { get; set; }

        public string OutputBaseDir { get; set; }

        public string TrainDataset { get; set; }

        public string TestDataset { get; set; }

        internal TaskKind MlTask { get; set; }

    }
}
