import type React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = "" }) => {
  return (
    <div
      className={`bg-card text-card-foreground rounded-lg border border-border p-6 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
};
