import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';

interface SidebarContextValue {
  collapsed: boolean;
  toggle: () => void;
  setCollapsed: (v: boolean) => void;
}

const SidebarContext = createContext<SidebarContextValue | null>(null);

const COLLAPSE_BREAKPOINT = 1100;
const STORAGE_KEY = 'sidebar-collapsed';

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsedState] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored !== null) return stored === 'true';
    return window.innerWidth < COLLAPSE_BREAKPOINT;
  });

  const setCollapsed = useCallback((v: boolean) => {
    setCollapsedState(v);
    localStorage.setItem(STORAGE_KEY, String(v));
  }, []);

  const toggle = useCallback(() => {
    setCollapsed(!collapsed);
  }, [collapsed, setCollapsed]);

  // Auto-collapse below breakpoint
  useEffect(() => {
    let prev = window.innerWidth;
    const onResize = () => {
      const w = window.innerWidth;
      if (prev >= COLLAPSE_BREAKPOINT && w < COLLAPSE_BREAKPOINT) setCollapsed(true);
      if (prev < COLLAPSE_BREAKPOINT && w >= COLLAPSE_BREAKPOINT) setCollapsed(false);
      prev = w;
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [setCollapsed]);

  return (
    <SidebarContext.Provider value={{ collapsed, toggle, setCollapsed }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar() {
  const ctx = useContext(SidebarContext);
  if (!ctx) throw new Error('useSidebar must be used within SidebarProvider');
  return ctx;
}
