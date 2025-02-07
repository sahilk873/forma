import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const video = formData.get('video') as Blob;

    // Send to Python backend
    const backendFormData = new FormData();
    backendFormData.append('video', video);

    const response = await fetch('http://localhost:8000/analyze', {  // Adjust URL as needed
      method: 'POST',
      body: backendFormData,
    });

    const analysis = await response.json();
    
    return NextResponse.json(analysis);
  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Failed to analyze video' },
      { status: 500 }
    );
  }
} 