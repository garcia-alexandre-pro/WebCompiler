﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class ScssTest
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
            File.Delete("../../artifacts/scss/test.css");
            File.Delete("../../artifacts/scss/test.min.css");
            File.Delete("../../artifacts/scss/relative.css");
            File.Delete("../../artifacts/scss/relative.min.css");
        }

        [TestMethod, TestCategory("SCSS")]
        public void CompileScss()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/scssconfig.json");
            CompilerResult first = result.First();
            Assert.IsTrue(File.Exists("../../artifacts/scss/test.css"));
            Assert.IsTrue(first.CompiledContent.Contains("/*# sourceMappingURL=data:"));
            Assert.IsTrue(result.ElementAt(1).CompiledContent.Contains("url(foo.png)"));
            Assert.IsTrue(result.ElementAt(1).CompiledContent.Contains("-webkit-animation"), "AutoPrefix");

            string sourceMap = DecodeSourceMap(first.CompiledContent);
            Assert.IsTrue(sourceMap.Contains("scss/test.scss"), "Source map paths");
        }

        [TestMethod, TestCategory("SCSS")]
        public void CompileScssError()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/scssconfigError.json");
            Assert.IsTrue(result.Count() == 1);
            Assert.IsTrue(result.ElementAt(0).HasErrors);
        }

        [TestMethod, TestCategory("SCSS")]
        public void AssociateExtensionSourceFileChangedTest()
        {
            IEnumerable<CompilerResult> result = _processor.SourceFileChanged(new FileInfo("../../artifacts/scssconfig.json").FullName, new FileInfo("../../artifacts/scss/_variables.scss").FullName, new DirectoryInfo("../../artifacts/").FullName);
            Assert.AreEqual(1, result.Count<CompilerResult>());
            Assert.IsTrue(File.Exists("../../artifacts/scss/test.css"));
        }

        [TestMethod, TestCategory("SCSS")]
        public void CommaListOfImportsSourcefileChanged()
        {
            IEnumerable<CompilerResult> result = _processor.SourceFileChanged(new FileInfo("../../artifacts/scssconfig.json").FullName, new FileInfo("../../artifacts/scss/sub/foo.scss").FullName, new DirectoryInfo("../../artifacts/").FullName);
            Assert.AreEqual(1, result.Count<CompilerResult>());
            Assert.IsTrue(File.Exists("../../artifacts/scss/test.css"));
        }

        [TestMethod, TestCategory("SCSS")]
        public void OtherExtensionTypeSourceFileChangedTest()
        {
            IEnumerable<CompilerResult> result = _processor.SourceFileChanged("../../artifacts/scssconfig.json", "scss/filewithinvalidextension.less", null);
            Assert.AreEqual(0, result.Count<CompilerResult>());
        }

        [TestMethod, TestCategory("SCSS")]
        public void MultiLineComments()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/scssconfig-no-sourcemap.json");
            Assert.IsTrue(result.First().CompiledContent.Contains("#test3"));
        }

        public static string DecodeSourceMap(string content)
        {
            Match match = Regex.Match(content, @"sourceMappingURL=data:application/json;.*base64,");

            if (match.Success)
            {
                int start = match.Index + match.Length;
                string map = content.Substring(start).Trim('*', '/');
                byte[] data = Convert.FromBase64String(map);
                return Encoding.UTF8.GetString(data);
            }

            return null;
        }
    }
}
