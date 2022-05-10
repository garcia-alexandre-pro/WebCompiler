﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class StylusTest
    {
        private ConfigFileProcessor _processor;

        [TestInitialize]
        public void Setup()
        {
            _processor = new ConfigFileProcessor();
        }

        [TestCleanup]
        public void Cleanup()
        {
            File.Delete("../../artifacts/stylus/output.css");
            File.Delete("../../artifacts/stylus/output.min.css");
        }

        [TestMethod, TestCategory("STYLUS")]
        public void CompileStylus()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/stylusconfig.json");
            Assert.IsTrue(File.Exists("../../artifacts/stylus/output.css"), "output doesn't exist");

            string sourceMap = ScssTest.DecodeSourceMap(result.First().CompiledContent);
            Assert.IsTrue(sourceMap.Contains("\"vendor.styl\""), "Source map paths");
        }

        [TestMethod, TestCategory("STYLUS")]
        public void CompileStylusWithError()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/stylusconfig.json");
            CompilerResult second = result.ElementAt(1);

            Assert.IsTrue(second.HasErrors, "Has no errors");
            Assert.IsTrue(second.Errors.First().ColumnNumber == 9, "Wrong column number");
        }
    }
}
