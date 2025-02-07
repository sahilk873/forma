'use client';

import Image from "next/image";
import VideoRecorder from "@/components/VideoRecorder";
import { useState } from "react";

interface AnalysisResult {
  issue: string;
  advice: string;
}

export default function Home() {
  const [requiresMirroring, setRequiresMirroring] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedExercise, setSelectedExercise] = useState<string>("");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (selectedFile) {
      setIsAnalyzing(true);
      const formData = new FormData();
      formData.append('video', selectedFile);
      formData.append('requiresMirroring', requiresMirroring.toString());
      formData.append('exercise', selectedExercise);

      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Analysis failed');
        }

        const result = await response.json();
        if (result.success) {
          setAnalysisResult(result.data);
        } else {
          throw new Error(result.error || 'Analysis failed');
        }
      } catch (error) {
        console.error('Error analyzing video:', error);
        alert('Failed to analyze video. Please try again.');
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center w-full max-w-4xl">
        <h1 className="text-3xl font-bold text-center">Forma</h1>
        
        <div className="w-full space-y-8">
          {/* Exercise Selection */}
          <div className="space-y-4">
            <div>
              <label htmlFor="exercise" className="block text-xl mb-2">Select Exercise</label>
              <select 
                id="exercise"
                value={selectedExercise}
                onChange={(e) => setSelectedExercise(e.target.value)}
                className="w-full p-3 border rounded-lg bg-background text-foreground"
              >
                <option value="">Choose an exercise...</option>
                <option value="squat">Squat</option>
                <option value="deadlift">Deadlift</option>
                <option value="bench-press">Bench Press</option>
                <option value="shoulder-press">Shoulder Press</option>
              </select>
            </div>

            <div>
              <label htmlFor="muscles" className="block text-xl mb-2">Target Muscles</label>
              <select 
                id="muscles"
                className="w-full p-3 border rounded-lg bg-background text-foreground"
              >
                <option value="">Select primary muscle group...</option>
                <option value="legs">Legs</option>
                <option value="back">Back</option>
                <option value="chest">Chest</option>
                <option value="shoulders">Shoulders</option>
                <option value="arms">Arms</option>
                <option value="core">Core</option>
              </select>
            </div>
          </div>

          {/* Video Upload/Record Section */}
          <div className="space-y-4">
            <h2 className="text-xl mb-4">Record or Upload Your Exercise Video</h2>
            
            {/* Video Recording Option */}
            <VideoRecorder />

            {/* Video Upload Option */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <h3 className="text-lg mb-4">Upload Video</h3>
              <input
                type="file"
                accept=".mov,.mp4"
                onChange={handleFileUpload}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-foreground file:text-background
                  hover:file:bg-[#383838]"
              />
              <div className="mt-4 flex items-center justify-center gap-2">
                <input
                  type="checkbox"
                  id="requiresMirroring"
                  checked={requiresMirroring}
                  onChange={(e) => setRequiresMirroring(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="requiresMirroring" className="text-sm text-gray-600">
                  Video needs to be un-mirrored (e.g., recorded with front-facing camera)
                </label>
              </div>
              <p className="text-sm text-gray-500 mt-2">
                Accepted formats: .mov, .mp4
              </p>
            </div>
          </div>

          {/* Analysis Button */}
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || !selectedExercise || isAnalyzing}
            className="w-full rounded-full bg-foreground text-background py-3 px-6 
              hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors
              disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Exercise Form'}
          </button>

          {/* Results Section */}
          <div className="mt-8 p-6 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <h2 className="text-xl mb-4">Analysis Results</h2>
            {analysisResult ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-lg">Issues Identified:</h3>
                  <p className="text-gray-600 dark:text-gray-400">{analysisResult.issue}</p>
                </div>
                <div>
                  <h3 className="font-semibold text-lg">Recommendations:</h3>
                  <p className="text-gray-600 dark:text-gray-400">{analysisResult.advice}</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-600 dark:text-gray-400">
                Upload your video and click analyze to see the results.
              </p>
            )}
          </div>
        </div>
      </main>

      <footer className="row-start-3 text-center text-sm text-gray-500">
        Forma - AI-powered exercise form analysis
      </footer>
    </div>
  );
}
