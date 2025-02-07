'use client';

import { useState, useRef, useCallback } from 'react';

interface VideoRecorderProps {
  onVideoSelect?: (blob: Blob) => void;
  selectedExercise?: string;
}

export default function VideoRecorder({ onVideoSelect, selectedExercise }: VideoRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startPreview = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          aspectRatio: { ideal: 16/9 }
        },
        audio: true 
      });
      
      streamRef.current = stream;
      
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.classList.remove('hidden');
        // Keep preview mirrored for user comfort
        videoPreviewRef.current.style.transform = 'scaleX(-1)';
      }
      
      setIsPreviewActive(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please make sure you have granted permission.');
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    const mediaRecorder = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = mediaRecorder;
    
    const chunks: BlobPart[] = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/mp4' });
      setVideoBlob(blob);
      
      // Create preview URL for the recorded video
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = null;
        videoPreviewRef.current.src = URL.createObjectURL(blob);
        videoPreviewRef.current.classList.remove('hidden');
      }
    };

    mediaRecorder.start();
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      setIsPreviewActive(false);
    }
  }, [isRecording]);

  const deleteRecording = useCallback(() => {
    if (videoPreviewRef.current) {
      videoPreviewRef.current.src = '';
      videoPreviewRef.current.srcObject = null;
      videoPreviewRef.current.classList.add('hidden');
    }
    setVideoBlob(null);
  }, []);

  const selectVideo = useCallback(async () => {
    if (videoBlob) {
      if (onVideoSelect) {
        onVideoSelect(videoBlob);  // Send the video blob to the parent component
      }
    }
  }, [videoBlob, onVideoSelect]);

  return (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
      <h3 className="text-lg mb-4">Record Video</h3>
      <video 
        ref={videoPreviewRef}
        className="w-full mb-4 hidden"
        autoPlay 
        playsInline 
        muted
      />
      <div className="space-x-4">
        {!isPreviewActive && !videoBlob && (
          <button
            onClick={startPreview}
            className="rounded-full bg-foreground text-background py-2 px-6 
              hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors"
          >
            Start Camera
          </button>
        )}
        {isPreviewActive && !isRecording && !videoBlob && (
          <button
            onClick={startRecording}
            className="rounded-full bg-foreground text-background py-2 px-6 
              hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors"
          >
            Start Recording
          </button>
        )}
        {isRecording && (
          <button
            onClick={stopRecording}
            className="rounded-full bg-red-500 text-white py-2 px-6 
              hover:bg-red-600 transition-colors"
          >
            Stop Recording
          </button>
        )}
        {videoBlob && (
          <>
            <button
              onClick={startPreview}
              className="rounded-full bg-foreground text-background py-2 px-6 
                hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors"
            >
              Record New
            </button>
            <button
              onClick={deleteRecording}
              className="rounded-full bg-red-500 text-white py-2 px-6 
                hover:bg-red-600 transition-colors"
            >
              Delete
            </button>
            <button
              onClick={selectVideo}
              className="rounded-full bg-green-500 text-white py-2 px-6 
                hover:bg-green-600 transition-colors"
            >
              Use This Video
            </button>
          </>
        )}
      </div>
    </div>
  );
} 