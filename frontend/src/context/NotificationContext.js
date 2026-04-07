import React, { createContext, useContext, useState, useCallback } from 'react';

const NotificationContext = createContext();

export function NotificationProvider({ children }) {
  const [notification, setNotification] = useState(null);

  const notify = useCallback((message, type = 'info', duration = 5000) => {
    setNotification({ message, type });
    if (duration > 0) {
      setTimeout(() => setNotification(null), duration);
    }
  }, []);

  const dismiss = useCallback(() => setNotification(null), []);

  return (
    <NotificationContext.Provider value={{ notify, dismiss }}>
      {children}
      {notification && (
        <div className={`notification notification-${notification.type}`} onClick={dismiss}>
          <span>{notification.message}</span>
          <button className="notification-close">&times;</button>
        </div>
      )}
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error('useNotification must be used within NotificationProvider');
  return {
    info: (msg) => ctx.notify(msg, 'info'),
    success: (msg) => ctx.notify(msg, 'success'),
    error: (msg) => ctx.notify(msg, 'error'),
    warning: (msg) => ctx.notify(msg, 'warning'),
    dismiss: ctx.dismiss,
  };
}
