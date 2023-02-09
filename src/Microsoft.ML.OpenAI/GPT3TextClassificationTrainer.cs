// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Newtonsoft.Json;
using OpenAI.GPT3;
using OpenAI.GPT3.Managers;
using OpenAI.GPT3.ObjectModels;
using OpenAI.GPT3.ObjectModels.ResponseModels.FineTuneResponseModels;

namespace Microsoft.ML.OpenAI
{
    public class GPT3TextClassificationTrainer : RowToRowTransformerBase, IEstimator<RowToRowTransformerBase>
    {
        private readonly string _labelColumnName;
        private readonly string _scoreColumnName;
        private readonly string _predictedLabelColumnName;
        private readonly string _textColumnName;
        private int _numberOfClasses;
        private readonly string _openAIKey;
        public string ModelID { get; set; }

        public GPT3TextClassificationTrainer(
            IHostEnvironment hostEnvironment,
            string labelColumnName,
            string scoreColumnName,
            string predictedLabelColumnName,
            string textColumnName,
            string openAIKey)
            : base(Contracts.CheckRef(hostEnvironment, nameof(hostEnvironment)).Register(nameof(GPT3TextClassificationTrainer)))
        {
            _labelColumnName = labelColumnName;
            _scoreColumnName = scoreColumnName;
            _predictedLabelColumnName = predictedLabelColumnName;
            _textColumnName = textColumnName;
            _openAIKey = openAIKey;
        }

        public RowToRowTransformerBase Fit(IDataView input)
        {
            using (var ch = Host.Start("Fine tune model"))
            {
                var openAIService = new OpenAIService(new OpenAiOptions
                {
                    ApiKey = _openAIKey,
                });
                //var texts = input.GetColumn<string>(_textColumnName);
                var labels = input.GetColumn<UInt32>(_labelColumnName);
                _numberOfClasses = labels.Distinct().Count();
                //Contracts.Assert(texts.Count() == labels.Count());

                //// create file to upload
                //var jsonL = Path.GetTempFileName();
                //ch.Trace($"create jsonL file: {jsonL}");
                //using (var stream = new StreamWriter(jsonL))
                //{
                //    foreach (var line in Enumerable.Zip(texts, labels))
                //    {
                //        var dict = new Dictionary<string, string>();
                //        dict["prompt"] = line.First;
                //        dict["completion"] = line.Second.ToString();
                //        var str = JsonConvert.SerializeObject(dict);
                //        ch.Trace($"add line: {str}");
                //        stream.WriteLine(str);
                //    }
                //    Host.CheckAlive();
                //}

                //// upload
                //var sampleFile = File.ReadAllBytes(jsonL);
                //var uploadFileResponse = openAIService.UploadFile("fine-tune", sampleFile, jsonL).Result;
                //if (uploadFileResponse.Successful)
                //{
                //    ch.Trace("upload file successfully");
                //}
                //else
                //{
                //    throw new ArgumentException(uploadFileResponse.Error?.Message);
                //}

                //// 
                //var createFineTuneResponse = openAIService.CreateFineTune(new global::OpenAI.GPT3.ObjectModels.RequestModels.FineTuneCreateRequest
                //{
                //    TrainingFile = uploadFileResponse.Id,
                //    Model = Models.Ada,
                //}).Result;

                //var listFineTuneEventsStream = openAIService.ListFineTuneEvents(createFineTuneResponse.Id, true).Result;
                //using var streamReader = new StreamReader(listFineTuneEventsStream);
                //while (!streamReader.EndOfStream)
                //{
                //    ch.Trace(streamReader.ReadLine());
                //}

                //FineTuneResponse retrieveFineTuneResponse;
                //do
                //{
                //    retrieveFineTuneResponse = openAIService.RetrieveFineTune(createFineTuneResponse.Id).Result;
                //    if (retrieveFineTuneResponse.Status == "succeeded" || retrieveFineTuneResponse.Status == "cancelled" || retrieveFineTuneResponse.Status == "failed")
                //    {
                //        ch.Trace($"Fine-tune Status for {createFineTuneResponse.Id}: {retrieveFineTuneResponse.Status}.");
                //        break;
                //    }

                //    ch.Trace($"Fine-tune Status for {createFineTuneResponse.Id}: {retrieveFineTuneResponse.Status}. Wait 10 more seconds");
                //    Task.Delay(10_000).Wait();
                //} while (true);

                ModelID = "ada:ft-personal-2023-02-09-05-27-43";
                ch.Trace($"Model ID {ModelID}");

                return this;
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var outColumns = inputSchema.ToDictionary(x => x.Name);
            var metaData = new List<SchemaShape.Column>();
            metaData.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                    TextDataViewType.Instance, false));
            // Get label column for score column annotations. Already verified it exists.
            inputSchema.TryFindColumn(_labelColumnName, out var labelCol);
            outColumns[_predictedLabelColumnName] = new SchemaShape.Column(_predictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.UInt32, true, new SchemaShape(metaData.ToArray()));

            outColumns[_scoreColumnName] = new SchemaShape.Column(_scoreColumnName, SchemaShape.Column.VectorKind.Vector,
                NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol)));

            return new SchemaShape(outColumns.Values);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
        {
            return new Mapper(Host, schema, this);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        private protected class Mapper : MapperBase
        {
            private protected readonly GPT3TextClassificationTrainer Parent;
            private protected readonly DataViewSchema.DetachedColumn DetachedLabelColumn;
            private protected readonly HashSet<int> InputColIndices;

            private static readonly FuncInstanceMethodInfo1<Mapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<Mapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);

            public Mapper(IHost host, DataViewSchema inputSchema, GPT3TextClassificationTrainer parent)
                : base(host, inputSchema, parent)
            {
                Parent = parent;
                InputColIndices = new HashSet<int>();
                if (inputSchema.TryGetColumnIndex(parent._textColumnName, out var col))
                {
                    InputColIndices.Add(col);
                }
                var labelColumn = inputSchema.GetColumnOrNull(Parent._labelColumnName);
                DetachedLabelColumn = new DataViewSchema.DetachedColumn(labelColumn!.Value);
            }

            private Delegate GetLabelAnnotations<T>(DataViewSchema.DetachedColumn labelCol)
            {
                return labelCol.Annotations.GetGetter<VBuffer<T>>(labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[2];
                var keyType = DetachedLabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType!.ItemType.RawType, DetachedLabelColumn);


                var meta = new DataViewSchema.Annotations.Builder();
                meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);
                meta.Add(AnnotationUtils.Kinds.SlotNames, keyType, getter);

                var labelBuilder = new DataViewSchema.Annotations.Builder();
                labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                info[0] = new DataViewSchema.DetachedColumn(Parent._predictedLabelColumnName, new KeyDataViewType(typeof(uint), Parent._numberOfClasses), labelBuilder.ToAnnotations());

                info[1] = new DataViewSchema.DetachedColumn(Parent._scoreColumnName, new VectorDataViewType(NumberDataViewType.Single, Parent._numberOfClasses), meta.ToAnnotations());
                return info;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                var ch = Host.Start("Make Getter");
                var textGetter = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[Parent._textColumnName]);
                var openAIService = new OpenAIService(new OpenAiOptions
                {
                    ApiKey = Parent._openAIKey,
                });
                ValueGetter<UInt32> classification = (ref UInt32 dst) =>
                {
                    ReadOnlyMemory<char> text = default;
                    textGetter.Invoke(ref text);
                    var result = openAIService.CreateCompletion(new global::OpenAI.GPT3.ObjectModels.RequestModels.CompletionCreateRequest
                    {
                        Model = Parent.ModelID,
                        MaxTokens = 1,
                        Prompt = text.ToString(),
                    }).Result;
                    var models = openAIService.ListModel().Result;
                    if (result.Successful)
                    {
                        dst = UInt32.Parse(result.Choices.First()?.Text!);
                    }
                };

                disposer = null;

                return classification;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => (activeOutput(0) || activeOutput(1)) && InputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                Parent.SaveModel(ctx);
            }
        }

    }
}
