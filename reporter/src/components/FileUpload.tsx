import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface Props {
  onFileLoaded: (text: string, filename: string) => void;
}

export default function FileUpload({ onFileLoaded }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];

      if (!file) {
        return;
      }

      const reader = new FileReader();

      reader.onload = () => onFileLoaded(reader.result as string, file.name);

      reader.readAsText(file);
    },
    [onFileLoaded],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/json': ['.json', '.jsonl'] },
    multiple: false,
  });

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] px-6">
      <h1 className="text-4xl font-bold tracking-tight mb-2">
        Pico<span className="text-accent">Router</span>
      </h1>

      <p className="text-muted mb-10 text-sm">Training Report Viewer</p>

      <div
        {...getRootProps()}
        className={`
          w-full max-w-xl border-2 border-dashed rounded-2xl p-14
          flex flex-col items-center justify-center gap-4 cursor-pointer
          transition-all duration-200
          ${isDragActive ? 'border-accent' : 'border-border hover:border-accent/50 hover:bg-surface2/50'}
        `}
      >
        <input {...getInputProps()} />

        <svg className="w-12 h-12 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>

        <>
          <p className="font-medium">
            Drop a <span className="font-mono text-accent">.jsonl</span> or{' '}
            <span className="font-mono text-accent">.json</span> training log
          </p>
          <p className="text-muted text-sm">or click to browse</p>
        </>
      </div>

      <p className="text-muted text-xs mt-6 max-w-md text-center leading-relaxed">
        Each line/entry should have <code className="font-mono text-accent/80">step</code>,{' '}
        <code className="font-mono text-accent/80">epoch</code>,{' '}
        <code className="font-mono text-accent/80">train_loss</code>,{' '}
        <code className="font-mono text-accent/80">lr</code>,{' '}
        <code className="font-mono text-accent/80">wall_clock</code>, and optionally{' '}
        <code className="font-mono text-accent/80">val_loss</code>.
      </p>
    </div>
  );
}
