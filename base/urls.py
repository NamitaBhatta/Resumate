from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.loginPage, name ="login"),
    path('logout/', views.logoutUser, name ="logout"),
    path('signup/', views.signupPage, name ="signup"),
    path('upload/', views.upload, name="upload"),

    
    path('',views.home, name="home"),
    path('aboutus/', views.aboutus, name="aboutus"),
    path('contactus/', views.aboutus, name="contactus"),
    path('view_resumes/', views.view_resumes, name='view_resumes'),
    path('result',views.result, name ="result")
]
