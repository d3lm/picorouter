import { useEffect, useState } from 'react';

export default function ThemeToggle() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') {
      return true;
    }

    return localStorage.getItem('picorouter-theme') !== 'light';
  });

  useEffect(() => {
    const root = document.documentElement;

    root.classList.add('theme-switching');

    if (dark) {
      root.classList.remove('light');
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
    }

    localStorage.setItem('picorouter-theme', dark ? 'dark' : 'light');

    window.dispatchEvent(new Event('picorouter-theme-change'));

    const timeoutId = window.setTimeout(() => {
      root.classList.remove('theme-switching');
    }, 50);

    return () => {
      window.clearTimeout(timeoutId);
      root.classList.remove('theme-switching');
    };
  }, [dark]);

  return (
    <div className="flex items-center gap-2">
      <span className="text-muted text-base leading-none">&#9790;</span>

      <button
        onClick={() => setDark((d) => !d)}
        role="switch"
        aria-checked={!dark}
        aria-label="Toggle light/dark theme"
        className={`
          relative w-[52px] h-7 rounded-full border border-border bg-surface2
          cursor-pointer transition-colors duration-200 shrink-0
        `}
      >
        <span
          className={`
            absolute top-[3px] left-[3px] w-5 h-5 rounded-full bg-accent
            transition-transform duration-200
            ${dark ? '' : 'translate-x-6'}
          `}
        />
      </button>

      <span className="text-muted text-base leading-none">&#9788;</span>
    </div>
  );
}
