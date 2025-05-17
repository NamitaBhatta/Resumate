from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.db.models import Q
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from .text_extraction_utils import extract_text_from_pdf
from django.core.files.base import ContentFile
from django.utils.text import slugify
# from .models import JobDescription, Resume
from .models import HRmanagerID, HRUser, Resume



# Create your views here.
def loginPage(request):
    # if user is loggedin they cant goto login page
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username').lower()
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User Doesnot Exist')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('upload')
        else:
            messages.error(request, 'Username or password doesnot exist')
    context ={}
    return render(request, 'base/login.html',context)


#@login_required(Login_url= 'login')
def logoutUser(request):
    logout(request)
    return redirect('home')


def signupPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirmPassword = request.POST.get('confirmPassword')
        if password and confirmPassword and password != confirmPassword:
            messages.error(request,"Passwords do not match.")
            return redirect('signup')
        else:
            user = User.objects.filter(username = username)
            if user.exists():
                messages.error(request, 'Username already taken')
                return redirect('signup')

            user = User.objects.create_user(username,email,password)
            user.set_password(password)
            user.save()
            messages.info(request, 'Account created successfully')
            login(request,user)
            return redirect('upload')
    return render(request, 'base/signup.html')


def home(request):
    return render(request, 'base/home.html')

def aboutus(request):
    return render(request, 'base/aboutus.html')

def contactus(request):
    return render(request, 'base/contactus.html')

def result(request):
    return render(request,'base/table.html')

#METHOD 4

# @login_required(login_url= 'login')
# def upload(request):
#     if request.method == 'POST':
#         user = request.user  # Get the currently logged-in user

#         # Assuming you have a form with 'jobdescription', 'skills', 'education', 'age', 'experience', 'resumes', and 'count' fields
#         hr_manager_id_instance = HRmanagerID.objects.create(HR_id=request.POST['HR_id'])

#         hr_user_instance = HRUser.objects.create(
#             user=user,
#             HR_id=hr_manager_id_instance,
#             jobdescription=request.FILES.get('jobdescription'),
#             skills=request.POST['skills'],
#             education=request.POST['education'],
#             age=request.POST['age'],
#             experience=request.POST['experience'],
#             count = request.POST['count']
#         )
        
#         resume_files = request.FILES.getlist('resumes')

#         for resume_file in resume_files:
#             # Create a Resume object associated with the HRUser instance
#             resume_obj = Resume.objects.create(user=user, file=resume_file)
#             hr_user_instance.resumes.add(resume_obj) 
#         hr_user_instance.save()

#         messages.success(request, 'Files uploaded successfully.')
#         return redirect('home')

#     return render(request, 'base/upload.html')

@login_required(login_url= 'login')
def upload(request):
    if request.method == 'POST':
        user = request.user  # Get the currently logged-in user

        # Assuming you have a form with 'jobdescription', 'skills', 'education', 'age', 'experience', 'resumes', and 'count' fields
        hr_manager_id_instance = HRmanagerID.objects.create(HR_id=request.POST['HR_id'])

        hr_user_instance = HRUser.objects.create(
            user=user,
            HR_id=hr_manager_id_instance,
            jobdescription=request.FILES.get('jobdescription'),
            skills=request.POST['skills'],
            education=request.POST['education'],
            age=request.POST['age'],
            experience=request.POST['experience'],
            count = request.POST['count'],

        )

        pdf_path = hr_user_instance.jobdescription.path
        extracted_text = extract_text_from_pdf(pdf_path)

        # Create a unique filename for the text content
        text_filename = slugify(hr_user_instance.jobdescription.name.split('.')[0]) + '.txt'

        #Save the extracted text to a text file and associate it with HRUser
        text_content_file = ContentFile(extracted_text.encode('utf-8'))
        hr_user_instance.jd_text_content.save(text_filename, text_content_file)

        resume_files = request.FILES.getlist('resumes')

        for resume_file in resume_files:
            # Create a Resume object associated with the HRUser instance
            resume_obj = Resume.objects.create(user=user, file=resume_file)
            hr_user_instance.resumes.add(resume_obj)
            resume_file_path = resume_obj.file.path

            resume_extracted_text = extract_text_from_pdf(resume_file_path)            
            resume_text_filename = slugify(resume_file.name.split('.')[0]) + '.txt'

            print(f"Resume Text Filename: {resume_text_filename}")
            
            # Save the extracted text to a text file and associate it with Resume
            resume_text_content_file = ContentFile(resume_extracted_text.encode('utf-8'))
            resume_obj.resume_text_content.save(resume_text_filename, resume_text_content_file)

            print(f"Resume Text Content Saved: {resume_obj.resume_text_content.name}")
        hr_user_instance.save()

        messages.success(request, 'Files uploaded successfully.')
        return redirect('home')

    return render(request, 'base/upload.html')



@login_required(login_url='login')
def view_resumes(request):
    user = request.user
    hr_user_instance = HRUser.objects.get(user=user)
    # resumes = hr_user_instance.resumes.all()
    resumes_with_text = []

    for resume in hr_user_instance.resumes.all():
        resume_text_content = resume.resume_text_content.name if resume.resume_text_content else None
        resumes_with_text.append({'resume': resume, 'resume_text_content': resume_text_content})

    return render(request, 'base/view_resumes.html', {'resumes_with_text': resumes_with_text})
    # return render(request, 'base/view_resumes.html', {'resumes': resumes})
