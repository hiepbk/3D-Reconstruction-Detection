#!/usr/bin/env python3
"""
Convert MP4 videos to GIF files for GitHub README display.

This script converts videos in the assets/ folder to optimized GIF files
that can be displayed directly in GitHub markdown.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_video_to_gif_ffmpeg(input_path, output_path, width=800, fps=10):
    """
    Convert video to GIF using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output GIF file
        width: Output width (height will be scaled proportionally)
        fps: Frames per second for GIF (lower = smaller file size)
    """
    # Use ffmpeg's palette-based method for better quality and smaller file size
    # Step 1: Generate palette
    palette_path = output_path.replace('.gif', '_palette.png')
    
    palette_cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,palettegen',
        '-y',  # Overwrite output file
        str(palette_path)
    ]
    
    print(f"Generating palette for {input_path.name}...")
    subprocess.run(palette_cmd, check=True)
    
    # Step 2: Convert to GIF using palette
    gif_cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-i', str(palette_path),
        '-filter_complex', f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse',
        '-y',  # Overwrite output file
        str(output_path)
    ]
    
    print(f"Converting {input_path.name} to GIF...")
    subprocess.run(gif_cmd, check=True)
    
    # Clean up palette file
    if os.path.exists(palette_path):
        os.remove(palette_path)
    
    print(f"✓ Created {output_path.name}")

def convert_video_to_gif_python(input_path, output_path, width=800, fps=10):
    """
    Convert video to GIF using Python libraries (moviepy/imageio).
    
    Args:
        input_path: Path to input video file
        output_path: Path to output GIF file
        width: Output width (height will be scaled proportionally)
        fps: Frames per second for GIF
    """
    try:
        from moviepy.editor import VideoFileClip
        import imageio
    except ImportError:
        print("Error: moviepy and imageio are required for Python conversion.")
        print("Install with: pip install moviepy imageio")
        sys.exit(1)
    
    print(f"Converting {input_path.name} to GIF using Python...")
    
    # Load video
    clip = VideoFileClip(str(input_path))
    
    # Resize if needed
    if clip.w > width:
        clip = clip.resize(width=width)
    
    # Set FPS
    clip = clip.set_fps(fps)
    
    # Convert to GIF
    clip.write_gif(str(output_path), fps=fps, program='imageio')
    
    print(f"✓ Created {output_path.name}")

def main():
    """Main conversion function."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    assets_dir = project_root / 'assets'
    
    if not assets_dir.exists():
        print(f"Error: assets directory not found at {assets_dir}")
        sys.exit(1)
    
    # Find all MP4 files
    video_files = list(assets_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"No MP4 files found in {assets_dir}")
        sys.exit(0)
    
    print(f"Found {len(video_files)} video file(s) to convert\n")
    
    # Check for ffmpeg first (preferred method)
    use_ffmpeg = check_ffmpeg()
    
    if use_ffmpeg:
        print("Using ffmpeg for conversion (better quality, smaller file size)\n")
        converter = convert_video_to_gif_ffmpeg
    else:
        print("ffmpeg not found. Using Python libraries (moviepy/imageio)\n")
        print("Note: ffmpeg produces better results. Install with: sudo apt-get install ffmpeg")
        print()
        converter = convert_video_to_gif_python
    
    # Convert each video
    for video_file in video_files:
        gif_file = assets_dir / f"{video_file.stem}.gif"
        
        try:
            converter(video_file, gif_file, width=800, fps=10)
        except Exception as e:
            print(f"✗ Error converting {video_file.name}: {e}")
            continue
    
    print(f"\n✓ Conversion complete! GIF files are in {assets_dir}")
    print("\nNext steps:")
    print("1. Update README.md to use .gif files instead of .mp4")
    print("2. Commit the GIF files to git")
    print("3. Update .gitignore to allow .gif files in assets/")

if __name__ == '__main__':
    main()

